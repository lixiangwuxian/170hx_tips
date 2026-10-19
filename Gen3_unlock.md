# CMP 170HX PCIe Gen3 开启指南

CMP 170HX 的 PCIe Gen3 需要三个条件同时成立：启动阶段允许 Gen3 的 fuse 软件覆盖、GPU 内部 Gen3 配置、驱动与上游端口完成链路训练。只设置 PCIe Target Link Speed，或者只看到最大能力为 Gen3，都不能证明链路已经建立。

本文适用于 Linux x86_64、GA100 CMP 170HX（本次实际设备 ID 为 `10de:2082`）、已能正常加载的 NVIDIA 开源内核模块。本文补充 fuse 覆盖的源码接入与验证步骤；原有 CMP 正常初始化支持仍是前提。参考驱动版本为 **615.71.09**。完整操作代码均列在正文中；不需要阅读作者的其他脚本。编译仍需要与驱动匹配的 NVIDIA 源码和系统编译工具。

# 验证范围

已有排障记录中，一张卡通过 RM 的 Gen1 → Gen2 → Gen3 顺序建立了 **8 GT/s x16** 链路；另一张参考卡也显示 Gen3 x16。另一次对不同卡的 Gen3 切换出现 GPU 丢失，不能据此认定该卡一定不支持 Gen3，也不能把它算作稳定通过。

这里的“建立”以实际 `LnkSta` 为准。长时间负载稳定性、每个主板插槽及每个驱动版本都需要单独验证。本文整理后的工具增加了输入检查，尚未在另一台机器上重新进行端到端硬件验证。

# 操作前确认

先停止目标 GPU 上的计算任务。链路切换可能中断 GPU，远程操作应具备重启和必要时断电恢复的条件。不要在使用同一 GPU 的桌面会话中直接切换。

以下命令中的 BDF 是显卡的 PCI 地址，必须替换成自己的实际地址。多卡逐张操作，不能用 nvidia-smi 的显示序号替代 BDF。

```bash
lspci -Dnn -d 10de:
nvidia-smi
cat /proc/driver/nvidia/version
modinfo -F version nvidia
modinfo -p nvidia | grep -E 'cmp_fuse|RegistryDwords'
BDF=0000:03:00.0
sudo lspci -s "$BDF" -vv
cat "/proc/driver/nvidia/gpus/$BDF/information"
```

前提是 GPU 初始化正常，已有 CUDA 或 OpenCL 程序能够运行。若已经存在 Xid、GPU 丢失或 BAR 映射错误，应先修复；Gen3 训练不能修复这些问题。

本节后面的训练工具为了保持与已成功的实现一致，限定 **PCI domain 0000、device 00、function 0**，即 `0000:xx:00.0`，普通独立 GPU、非 SLI。其他 PCI 拓扑不能直接套用工具，需要扩展 PCI 匹配逻辑。这是工具的适用边界，不代表硬件只能用于该地址。

# 驱动必须完成的配置

## Fuse 覆盖与驱动参数

参考实现用 `cmp_fuse_gen3=0` 在 probe 阶段清除 Gen3 禁止状态，涉及 BAR0 偏移 `0x820250` 与 `0x820580`。这是易失的软件覆盖；本文不进行物理熔丝烧写或 VBIOS 刷写。其他 fuse 解锁实现必须达到同等效果，参数名称未必相同。

参考驱动的参数组合如下：

```text
options nvidia NVreg_RegistryDwords="RmForceEnableGen2=1;RMPcieLinkSpeed=0x5;CmpGen3=1" cmp_fuse_gen3=0
```

`RMPcieLinkSpeed=0x5` 是该实现用于开放 Gen3 的位字段，不是“第五代”。`CmpGen3` 是自定义补丁读取的键，原版驱动不会因为写入这个名字就自动实现本文的寄存器操作。

先查找现有配置并备份相应文件：

```bash
sudo grep -RnsE 'NVreg_RegistryDwords|cmp_fuse|CmpGen3|RMPcieLinkSpeed' /etc/modprobe.d /usr/lib/modprobe.d 2>/dev/null
modprobe -c | grep '^options nvidia '
```

将上述键合并到现有配置中，同一键只保留一个有效值。保留原有 SM/TPC 屏蔽、显存和 BAR 参数，尤其不能复制其他卡的 TPC 屏蔽结果。检查 `NVreg_RegistryDwordsPerDevice` 是否有针对该卡的冲突值，例如 `CmpGen3=0`。

### 1. 确认模块实际提供 fuse 覆盖功能

`cmp_fuse_gen3` 不是 NVIDIA 原版参数。先检查准备安装的模块；驱动已加载时还应检查当前模块的参数文件：

```bash
modinfo -p nvidia | grep -E '^cmp_fuse_(gen3|bdf):'
# 仅在 nvidia 已加载时执行；这是当前运行模块，而不是磁盘上的候选模块
ls /sys/module/nvidia/parameters/cmp_fuse_{gen3,bdf}
```

如果缺少这些参数，不能只向 modprobe 配置里添加 `cmp_fuse_gen3=0`。先完成下面的源码接入，再与后续 Gen3 代码一起编译安装。已有同名参数和 `nv_cmp170hx_fuse_override()` 的源码不要重复粘贴；核对函数是否确实在 probe 阶段被调用。

### 2. 补齐 probe 阶段的 fuse 覆盖实现

文件是 `kernel-open/nvidia/nv-pci.c`。以下代码取自本次使用的 615.71.09 源码中的 fuse 覆盖块，仅省略无关的 BAR1 容量参数声明。把它放在现有头文件之后、`nv_pci_probe()` 定义之前；已有的相同头文件不必重复添加。

这里保留完整参数表和函数，是因为开启 `EN_SW_OVERRIDE` 前，需要先把表内已有的 GPC/TPC 等 OPT 值填入对应 `_CP` 影子寄存器，避免把未要求修改的项目换成旧的影子值。**本指南只设置 `cmp_fuse_gen3=0`；不要为了开启 Gen3 再设置 `cmp_fuse_gpc=0`、`cmp_fuse_tpc*=0` 或 `cmp_fuse_flr=1`。** 其余参数保持默认值或原来经过验证的配置。代码包含其他功能，不代表这些分支都属于本指南的验证范围。

<details>
<summary>展开：完整 probe fuse 覆盖代码</summary>

```c
#include <linux/io.h>
#include <linux/moduleparam.h>

/*
 * cmpunlocker: early fuse SW override (EXPERIMENTAL, off by default).
 *
 * Runs in nv_pci_probe() right after BAR0 is claimed -- before RM init and
 * GSP boot -- so anything that consumes the OPT_* fuses during driver
 * bring-up sees the override instead of the burned value.
 * Equivalent to `ga100_rir --sw-enable` + `--sw-write <reg> <val>`.
 *
 * Two kinds of registers:
 *  - floorsweep masks with a _CP shadow (gpc/fbp/fbpa/fbio/l2/nvlink):
 *    written to _CP with EN_SW_OVERRIDE=1;
 *  - fuses without a _CP shadow (gen3/ecc/ecc_ovr*): the register itself is
 *    written directly. This may be PLM-blocked this early; the readback
 *    in dmesg shows whether it stuck.
 *
 * Every value is VOLATILE: a GPU reset clears it. -1 (default) = leave that
 * register alone. cmp_fuse_bdf limits the override to one GPU. Example:
 *   options nvidia cmp_fuse_bdf=0000:02:00.0 cmp_fuse_gpc=0 cmp_fuse_gen3=0
 */
#define CMP_FUSE_EN_SW_OVERRIDE         0x00820040
#define CMP_FUSE_DISABLE_SW_OVERRIDE    0x00820084
/* must cover the highest register in cmp_fuse_regs (0x82382C) */
#define CMP_FUSE_MAP_SIZE               0x00824000

static char *cmp_fuse_bdf  = "";
static int cmp_fuse_gpc    = -1;
static int cmp_fuse_tpc0   = -1;
static int cmp_fuse_tpc1   = -1;
static int cmp_fuse_tpc2   = -1;
static int cmp_fuse_tpc3   = -1;
static int cmp_fuse_tpc4   = -1;
static int cmp_fuse_tpc5   = -1;
static int cmp_fuse_tpc6   = -1;
static int cmp_fuse_tpc7   = -1;
static int cmp_fuse_fbp    = -1;
static int cmp_fuse_fbpa   = -1;
static int cmp_fuse_fbio   = -1;
static int cmp_fuse_l2     = -1;
static int cmp_fuse_nvlink = -1;
static int cmp_fuse_gen3   = -1;
static int cmp_fuse_ecc    = -1;
static int cmp_fuse_ecc_ovr  = -1;
static int cmp_fuse_ecc_ovr2 = -1;
static int cmp_fuse_ecc_ovr3 = -1;
static int cmp_fuse_flr    = 0;
module_param(cmp_fuse_bdf,    charp, 0444);
module_param(cmp_fuse_gpc,    int, 0444);
module_param(cmp_fuse_tpc0,   int, 0444);
module_param(cmp_fuse_tpc1,   int, 0444);
module_param(cmp_fuse_tpc2,   int, 0444);
module_param(cmp_fuse_tpc3,   int, 0444);
module_param(cmp_fuse_tpc4,   int, 0444);
module_param(cmp_fuse_tpc5,   int, 0444);
module_param(cmp_fuse_tpc6,   int, 0444);
module_param(cmp_fuse_tpc7,   int, 0444);
module_param(cmp_fuse_fbp,    int, 0444);
module_param(cmp_fuse_fbpa,   int, 0444);
module_param(cmp_fuse_fbio,   int, 0444);
module_param(cmp_fuse_l2,     int, 0444);
module_param(cmp_fuse_nvlink, int, 0444);
module_param(cmp_fuse_gen3,   int, 0444);
module_param(cmp_fuse_ecc,    int, 0444);
module_param(cmp_fuse_ecc_ovr,  int, 0444);
module_param(cmp_fuse_ecc_ovr2, int, 0444);
module_param(cmp_fuse_ecc_ovr3, int, 0444);
module_param(cmp_fuse_flr,    int, 0444);
MODULE_PARM_DESC(cmp_fuse_bdf,    "CMP 170HX: only apply fuse overrides to this PCI BDF (empty = all)");
MODULE_PARM_DESC(cmp_fuse_gpc,    "CMP 170HX: OPT_GPC_DISABLE_CP override at probe (-1 = off)");
MODULE_PARM_DESC(cmp_fuse_fbp,    "CMP 170HX: OPT_FBP_DISABLE_CP override at probe (-1 = off)");
MODULE_PARM_DESC(cmp_fuse_fbpa,   "CMP 170HX: OPT_FBPA_DISABLE_CP override at probe (-1 = off)");
MODULE_PARM_DESC(cmp_fuse_fbio,   "CMP 170HX: OPT_FBIO_DISABLE_CP override at probe (-1 = off)");
MODULE_PARM_DESC(cmp_fuse_l2,     "CMP 170HX: OPT_ROP_L2_DISABLE_CP override at probe (-1 = off)");
MODULE_PARM_DESC(cmp_fuse_nvlink, "CMP 170HX: OPT_NVLINK_DISABLE_CP override at probe (-1 = off)");
MODULE_PARM_DESC(cmp_fuse_gen3,   "CMP 170HX: OPT_DISABLE_GEN3_SPEED + OPT_PCIE_BOOT_GEN3_DIS direct write at probe (-1 = off)");
MODULE_PARM_DESC(cmp_fuse_ecc,    "CMP 170HX: OPT_ECC_EN direct write at probe (-1 = off)");
MODULE_PARM_DESC(cmp_fuse_ecc_ovr,  "CMP 170HX: FEATURE_OVERRIDE_ECC (0x82380c) direct write at probe, e.g. 0x999999 (-1 = off)");
MODULE_PARM_DESC(cmp_fuse_ecc_ovr2, "CMP 170HX: 0x823810 ECC feature override direct write at probe, e.g. 0x3ffebf (-1 = off)");
MODULE_PARM_DESC(cmp_fuse_ecc_ovr3, "CMP 170HX: 0x82382c ECC feature override direct write at probe, e.g. 0xf (-1 = off)");

/* cp == 0: no _CP shadow, write the OPT register directly */
static const struct
{
    const char *name;
    int        *param;
    NvU32       cp;
    NvU32       opt;
    NvU32       mask;
} cmp_fuse_regs[] = {
    { "gpc",       &cmp_fuse_gpc,    0x0082022C, 0x00820350, 0x000000FF },
    { "tpc0",      &cmp_fuse_tpc0,   0x00820230, 0x00820760, 0x000000FF },
    { "tpc1",      &cmp_fuse_tpc1,   0x00820504, 0x00820764, 0x000000FF },
    { "tpc2",      &cmp_fuse_tpc2,   0x008205F4, 0x00820768, 0x000000FF },
    { "tpc3",      &cmp_fuse_tpc3,   0x008205A0, 0x0082076c, 0x000000FF },
    { "tpc4",      &cmp_fuse_tpc4,   0x008204C4, 0x00820770, 0x000000FF },
    { "tpc5",      &cmp_fuse_tpc5,   0x008204CC, 0x00820774, 0x000000FF },
    { "tpc6",      &cmp_fuse_tpc6,   0x00820B8C, 0x00820778, 0x000000FF },
    { "tpc7",      &cmp_fuse_tpc7,   0x00820B88, 0x0082077c, 0x000000FF },
    { "gen3",      &cmp_fuse_gen3,   0,          0x00820250, 0x00000001 },
    { "gen3_boot", &cmp_fuse_gen3,   0,          0x00820580, 0x00000001 },
    { "ecc",       &cmp_fuse_ecc,    0,          0x00820228, 0x00000001 },
    { "ecc_ovr",   &cmp_fuse_ecc_ovr,  0,        0x0082380C, 0x7FFFFFFF },
    { "ecc_ovr2",  &cmp_fuse_ecc_ovr2, 0,        0x00823810, 0x7FFFFFFF },
    { "ecc_ovr3",  &cmp_fuse_ecc_ovr3, 0,        0x0082382C, 0x7FFFFFFF },
    { "nvlink",    &cmp_fuse_nvlink,  0x00820688, 0x00820684, 0x0000FFFF },
};

static void
nv_cmp170hx_fuse_override(struct pci_dev *pci_dev, NvU8 bar)
{
    void __iomem *bar0;
    NvBool requested = NV_FALSE;
    unsigned int i;

    if ((pci_dev->vendor != 0x10de) ||
        ((pci_dev->device != 0x20c2) && (pci_dev->device != 0x2082)))
        return;

    if ((cmp_fuse_bdf != NULL) && (cmp_fuse_bdf[0] != '\0') &&
        (strcmp(cmp_fuse_bdf, pci_name(pci_dev)) != 0))
        return;

    for (i = 0; i < ARRAY_SIZE(cmp_fuse_regs); i++)
    {
        /* an access past the ioremap window oopses inside probe and hangs boot */
        if ((cmp_fuse_regs[i].opt + 4 > CMP_FUSE_MAP_SIZE) ||
            (cmp_fuse_regs[i].cp + 4 > CMP_FUSE_MAP_SIZE))
        {
            dev_err(&pci_dev->dev, "CMP fuse: %s is outside the BAR0 map; refusing all overrides\n",
                    cmp_fuse_regs[i].name);
            return;
        }
        if (*cmp_fuse_regs[i].param >= 0)
            requested = NV_TRUE;
    }
    if (!requested)
        return;

    bar0 = ioremap(NV_PCI_RESOURCE_START(pci_dev, bar), CMP_FUSE_MAP_SIZE);
    if (bar0 == NULL)
    {
        dev_err(&pci_dev->dev, "CMP fuse: cannot map BAR0; skipping override\n");
        return;
    }

    if (readl(bar0 + CMP_FUSE_DISABLE_SW_OVERRIDE) & 1)
    {
        dev_err(&pci_dev->dev, "CMP fuse: disable_sw_override fuse is burned; skipping\n");
        goto out;
    }

    /*
     * Seed every known _CP with the current fused value before flipping
     * EN_SW_OVERRIDE, so registers we were not asked to change keep their
     * fused value instead of whatever _CP held after reset.
     */
    for (i = 0; i < ARRAY_SIZE(cmp_fuse_regs); i++)
    {
        NvU32 fused;

        if (cmp_fuse_regs[i].cp == 0)
            continue;
        fused = readl(bar0 + cmp_fuse_regs[i].opt) & cmp_fuse_regs[i].mask;
        writel(fused, bar0 + cmp_fuse_regs[i].cp);
    }

    writel(1, bar0 + CMP_FUSE_EN_SW_OVERRIDE);
    (void)readl(bar0 + CMP_FUSE_EN_SW_OVERRIDE);

    for (i = 0; i < ARRAY_SIZE(cmp_fuse_regs); i++)
    {
        NvU32 mask = cmp_fuse_regs[i].mask;
        NvU32 val, fused, opt;

        if (*cmp_fuse_regs[i].param < 0)
            continue;

        val   = (NvU32)*cmp_fuse_regs[i].param & mask;
        fused = readl(bar0 + cmp_fuse_regs[i].opt) & mask;

        if (cmp_fuse_regs[i].cp == 0)
        {
            NvU32 raw = readl(bar0 + cmp_fuse_regs[i].opt);
            writel((raw & ~mask) | val, bar0 + cmp_fuse_regs[i].opt);
            opt = readl(bar0 + cmp_fuse_regs[i].opt) & mask;
            dev_warn(&pci_dev->dev,
                     "CMP fuse: %s fused=0x%x write=0x%x opt=0x%x (direct)%s\n",
                     cmp_fuse_regs[i].name, fused, val, opt,
                     (opt != val) ? " (DID NOT STICK)" : "");
        }
        else
        {
            NvU32 cp, raw;

            /*
             * Dual write: OPT direct + _CP shadow. The OPT layer is what
             * STATUS_OPT_* re-evaluates from (runtime-proven: writing
             * OPT_GPC_DISABLE=0 flips STATUS_OPT_GPC immediately); the _CP
             * shadow keeps cross-checks happy. Direct OPT writes stick at
             * probe time (verified with the gen3 pair).
             */
            raw = readl(bar0 + cmp_fuse_regs[i].opt);
            writel((raw & ~mask) | val, bar0 + cmp_fuse_regs[i].opt);
            writel(val, bar0 + cmp_fuse_regs[i].cp);
            cp  = readl(bar0 + cmp_fuse_regs[i].cp) & mask;
            opt = readl(bar0 + cmp_fuse_regs[i].opt) & mask;
            dev_warn(&pci_dev->dev,
                     "CMP fuse: %s fused=0x%x write=0x%x cp=0x%x opt=0x%x (dual)%s\n",
                     cmp_fuse_regs[i].name, fused, val, cp, opt,
                     (opt != val) ? " (OPT DID NOT STICK)" :
                     (cp != val) ? " (CP DID NOT STICK)" : "");
        }
    }

    /*
     * cmpunlocker: cmp_fuse_flr=1 -- floorsweep masks are latched at GPU
     * reset; request a function-level reset after the _CP override so the
     * fuse block re-evaluates the SW override. Diagnostic: read back
     * STATUS_OPT_GPC (0x820c1c) before/after to see whether the latch
     * cleared (GA100: RM consumes STATUS_OPT_GPC, not OPT_GPC_DISABLE).
     */
    if (cmp_fuse_flr)
    {
        NvU32 statusBefore = readl(bar0 + 0x00820c1c);
        NvU32 cpGpc = readl(bar0 + 0x0082022c);
        dev_warn(&pci_dev->dev,
                 "CMP fuse: pre-FLR STATUS_OPT_GPC=0x%08x GPC_CP=0x%08x\n",
                 statusBefore, cpGpc);
        iounmap(bar0);

        if (pci_reset_function(pci_dev) == 0)
        {
            void __iomem *bar0b;
            bar0b = ioremap(NV_PCI_RESOURCE_START(pci_dev, bar), CMP_FUSE_MAP_SIZE);
            if (bar0b != NULL)
            {
                dev_warn(&pci_dev->dev,
                         "CMP fuse: post-FLR STATUS_OPT_GPC=0x%08x OPT_GPC=0x%08x EN_SW_OVERRIDE=%u\n",
                         readl(bar0b + 0x00820c1c),
                         readl(bar0b + 0x00820350),
                         readl(bar0b + CMP_FUSE_EN_SW_OVERRIDE) & 1);
                iounmap(bar0b);
            }
            else
                dev_err(&pci_dev->dev, "CMP fuse: post-FLR BAR0 remap failed\n");
        }
        else
            dev_err(&pci_dev->dev, "CMP fuse: FLR failed\n");
        return;
    }

    dev_warn(&pci_dev->dev, "CMP fuse: EN_SW_OVERRIDE=%u (pre-RM/GSP)\n",
             readl(bar0 + CMP_FUSE_EN_SW_OVERRIDE) & 1);
out:
    iounmap(bar0);
}
```

</details>

然后在 `nv_pci_probe()` 内，BAR0 已验证并成功申请之后、原有 `NV_KZALLOC(nvl, ...)` 之前加入一次调用。保留周围原有错误处理，不要把调用移到 BAR0 尚未可访问的位置，也不要等到 RM/GSP 初始化完成后才执行：

```c
    /* BAR0 validation and request_mem_region have succeeded above. */
    nv_cmp170hx_fuse_override(pci_dev, regs_bar_index);

    NV_KZALLOC(nvl, sizeof(nv_linux_state_t));
```

在这个实现中，`0x00820084` 的 bit 0 若禁止软件覆盖，函数会记录日志并跳过；不会尝试烧写熔丝。正常路径先初始化已知 `_CP`，再设置 `0x00820040` 的 `EN_SW_OVERRIDE`，最后清除 `0x00820250` 与 `0x00820580` 的 bit 0，并立即读回。这两个 Gen3 项走直接 OPT 写入路径，不是通过 TPC 位图实现。

`cmp_fuse_gen3=-1` 是默认“不修改”；`0` 表示把 Gen3 禁止位写成 0，即允许 Gen3，不是关闭覆盖功能。设置参数不会改写已经加载的模块：修改后需要按后文重新构建/加载或重启。GPU 复位后也必须重新应用并验证，不能当作永久解锁。

### 3. 为目标卡设置参数并重新加载

仅对一张卡进行操作时，可以把完整 BDF 限制加入已有参数行，例如：

```text
options nvidia NVreg_RegistryDwords="RmForceEnableGen2=1;RMPcieLinkSpeed=0x5;CmpGen3=1" cmp_fuse_bdf=0000:03:00.0 cmp_fuse_gen3=0
```

上面只是 Gen3 相关参数的合并示例，不要用它覆盖现有整行，丢掉 SM/TPC、BAR 等配置。BDF 必须以本机枚举结果为准。多张受支持的 CMP 170HX 都需要覆盖时，可不指定 `cmp_fuse_bdf`，使用默认空字符串；代码仍只匹配 `10de:20c2/2082`。本文硬件验证对象是 `2082`，不要因为代码包含 `20c2` 就宣称后者也已验证。

`cmp_fuse_bdf` 本身不是 ECID 匹配：换槽后必须更新或移除旧限制。若另有 ECID 加载器，应让它继续负责各卡 TPC 位图；不要用相同的 TPC 屏蔽值替代它。保留虚拟机直通卡的 VFIO 绑定，使用本机已有的加载保护流程；`modprobe -r` 可能连带卸载软依赖，而 `--ignore-install` 或直接 `insmod` 可能绕过自定义加载器。

完成后文的 GSP 后置 Gen3 代码、编译安装和启动镜像处理，再重启进入新模块。已有覆盖实现且仅改参数时，也需要在停止 GPU 客户端后按原有流程重新加载，或重启；无需为了读取状态在线反复写寄存器。

### 4. 验证覆盖确实执行，并检查初始化后的读回值

重启后检查当前模块参数和本次启动日志：

```bash
BDF=0000:03:00.0
cat /sys/module/nvidia/parameters/cmp_fuse_gen3
cat /sys/module/nvidia/parameters/cmp_fuse_bdf
sudo journalctl -k -b --no-pager | grep -E "${BDF}.*CMP fuse:|CMP fuse:.*${BDF}"
```

目标卡应出现类似 `CMP fuse: gen3 ... write=0x0 opt=0x0 (direct)`、`CMP fuse: gen3_boot ... write=0x0 opt=0x0 (direct)` 和 `EN_SW_OVERRIDE=1 (pre-RM/GSP)` 的记录。若出现 `DID NOT STICK`、映射失败或覆盖被禁止，应先解决覆盖问题；仅看到参数文件里的 `0` 不算写入成功。缺少日志也需要核对实际加载的模块、BDF 限制和调用位置。

probe 日志只证明那个时刻的状态。GSP/驱动后续初始化可能改变寄存器，因此 GPU 正常初始化后，可以用以下只读检查核对当前值；它不写寄存器，也不切换链路：

```bash
sudo python3 - "$BDF" <<'PYREAD'
import mmap, os, pathlib, struct, sys
bdf = sys.argv[1]
p = pathlib.Path('/sys/bus/pci/devices') / bdf
if (p / 'vendor').read_text().strip() != '0x10de' or (p / 'device').read_text().strip() != '0x2082':
    raise SystemExit('Not the validated 10de:2082 device')
if not (p / 'driver').exists() or (p / 'driver').resolve().name != 'nvidia':
    raise SystemExit('Run after successful NVIDIA initialization')
regs = [(0x820040, 'EN_SW_OVERRIDE'), (0x820084, 'DISABLE_SW_OVERRIDE'),
        (0x820250, 'OPT_DISABLE_GEN3_SPEED'),
        (0x820580, 'OPT_PCIE_BOOT_GEN3_DIS'),
        (0x82057c, 'OPT_GEN23 diagnostic only')]
fd = os.open(p / 'resource0', os.O_RDONLY | os.O_SYNC)
try:
    with mmap.mmap(fd, 0x824000, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ) as m:
        values = {off: struct.unpack_from('<I', m, off)[0] for off, _ in regs}
        for off, name in regs:
            print(f'{name:26s} 0x{off:08x} = 0x{values[off]:08x}')
        if any(v == 0xffffffff or v & 0xffff0000 == 0xbadf0000 for v in values.values()):
            raise SystemExit('Unreadable register; do not interpret as an override result')
        if not (values[0x820040] & 1) or values[0x820084] & 1 or values[0x820250] & 1 or values[0x820580] & 1:
            raise SystemExit('Expected Gen3 fuse override state is not present')
        print('Expected Gen3 fuse override state is present; link training is still a separate check')
finally:
    os.close(fd)
PYREAD
```

这里的 `OPT_GEN23`（`0x0082057c`）只用于诊断。它与上面两个 Gen3 禁止位不是同一寄存器，不应在缺少对应驱动时序验证时，把“再清掉这一位”当作通用补丁。本次曾试加该项，未解决冷启动问题，已撤销；不能把这个试验步骤写成已验证的必需操作。

即使两个禁止位均读回 0，也仍需后面的 GSP Gen3 配置、RM 训练以及 `LnkSta`/负载/重启验收。**fuse 覆盖成功不等于已经建立稳定 Gen3 链路。**

## 只有 fuse 覆盖时需要补充的驱动代码

如果现有驱动已经实现了下列 Gen3 操作，不要重复添加。否则，需要在已有 fuse 补丁的 NVIDIA 开源驱动源码上加入此逻辑。仅有 fuse 覆盖的二进制模块，无法靠配置文件补出缺少的驱动代码。

参考插入位置是 `src/nvidia/src/kernel/gpu/gsp/arch/turing/kernel_gsp_tu102.c` 的 `kgspBootstrap_TU102`：正常 BooterLoad 返回成功后、函数最终错误处理之前。必须处于 GSP 启动后的正确时序，不能提前到 probe 阶段。以下代码块整理了参考实现中的 Gen3 操作；不同版本若函数或初始化流程变化，应先适配，不能机械插入。

```c
/* Inside kgspBootstrap_TU102, after successful BooterLoad.
 * Requires the existing fuse-override patch and normal CMP initialization.
 * CmpGen3=0 skips this added Gen3 block; it is not a runtime rollback.
 */
if (status == NV_OK && (pGpu->idInfo.PCIDeviceID >> 16) == 0x2082)
{
    NvU32 enableGen3 = 0;
    (void)osReadRegistryDword(pGpu, "CmpGen3", &enableGen3);
    if (enableGen3 == 1)
    {
        NvU32 v, dlmgr, i;
        v = GPU_REG_RD32(pGpu, 0x0008841c);
        v = (v | (1U << 11) | (1U << 13)) &
            ~((1U << 12) | (1U << 14) | 0xc0000000U);
        GPU_REG_WR32(pGpu, 0x0008841c, v);
        v = GPU_REG_RD32(pGpu, 0x0008c2c0);
        GPU_REG_WR32(pGpu, 0x0008c2c0, v & ~((1U << 2) | (1U << 3)));
        v = GPU_REG_RD32(pGpu, 0x000880a8);
        GPU_REG_WR32(pGpu, 0x000880a8, (v & ~0xfU) | 3U);
        v = GPU_REG_RD32(pGpu, 0x0008c1c0);
        GPU_REG_WR32(pGpu, 0x0008c1c0, (v & ~0x000f0000U) | (3U << 16));
        v = GPU_REG_RD32(pGpu, 0x00088610);
        GPU_REG_WR32(pGpu, 0x00088610, v | 0x1000U);
        v = GPU_REG_RD32(pGpu, 0x0008c040);
        /* Internal rate encoding: 1 means 8 GT/s here. */
        GPU_REG_WR32(pGpu, 0x0008c040, (v & ~0x000c0000U) | (1U << 18));
        dlmgr = GPU_REG_RD32(pGpu, 0x0008b8c0);
        GPU_REG_WR32(pGpu, 0x0008b8c0, dlmgr | 4U);
        for (i = 0; i < 100000; i++)
        {
            v = GPU_REG_RD32(pGpu, 0x0008c040);
            if (v == 0xffffffffU || (v & 0x1fU) == 0) break;
        }
        if (v != 0xffffffffU && (v & 0x1fU) == 0)
        {
            GPU_REG_WR32(pGpu, 0x0008c040, (v & ~0xfU) | 1U);
            for (i = 0; i < 100000; i++)
            {
                v = GPU_REG_RD32(pGpu, 0x0008c040);
                if (v == 0xffffffffU || (v & 0x1fU) == 0) break;
            }
        }
        if (v != 0xffffffffU)
            GPU_REG_WR32(pGpu, 0x0008b8c0, dlmgr);
        if (v != 0xffffffffU && (v & 0x1fU) == 0)
            GPU_REG_WR32(pGpu, 0x0008872c, 8U);
        /* Timeout or inaccessible registers must not be reported as success.
         * Actual success is checked later using PCIe LnkSta.
         */
    }
}
```

此块限定 `2082`，不据此推广到其他 NVIDIA 型号。它不是可独立运行的用户态程序，也不是对所有驱动版本都适用的补丁文件。保留已有正常 CMP 初始化逻辑。本整理版增加了等待成功检查，尚未单独完成驱动编译及硬件回归验证。

## 编译安装及重启

需要当前运行内核对应的开发头文件、C/C++ 编译器、make，以及与 NVIDIA 用户态库和 GSP 固件完全匹配的开源模块源码。以已有 fuse 补丁的源码树为工作目录，不要用全新的原版源码覆盖掉 fuse 功能。

源码直接安装的示例：

```bash
# 在已打补丁、版本匹配的 open-gpu-kernel-modules 源码根目录执行
make -j"$(nproc)" modules
# 只有上一步完全成功才继续；事先保留旧模块或可启动的旧内核
sudo make modules_install
sudo depmod -a
```

若原安装使用 DKMS 或发行版软件包，应通过原渠道重建安装，避免直接安装与包管理相互覆盖。启用 Secure Boot 的系统需要按原方式签名并让系统信任新模块，否则重启后模块可能被拒绝加载。

当 initramfs 包含 NVIDIA 模块或相关配置时，还要更新它。Debian/Ubuntu 常用 `sudo update-initramfs -u -k "$(uname -r)"`；采用 dracut 的系统常用 `sudo dracut -f`。选择自己的发行版对应方式，不要两条都执行。若原方案刻意不将 NVIDIA 放入 initramfs，应保持该启动设计。

保存工作后重启。重启后再次核对 `/proc/driver/nvidia/version`、`nvidia-smi`、内核日志，并确认新模块实际加载。仅 `modinfo` 显示新版本不代表当前运行的模块已更新。

# 完整 RM 训练工具

将下面完整内容保存为 `set_pcie_speed.c`。它通过 NVIDIA RM 接口请求切换速率，让驱动执行训练。输入 `1/2/3` 分别代表 Gen1/Gen2/Gen3，`0` 只查询映射；**不能输入 2500、5000、8000**。

设备 minor 从 `/proc/driver/nvidia/gpus/BDF/information` 取得，RM deviceInstance 则通过 GPU ID 查询取得，二者不能混用。NVIDIA 的 [GPU 控制接口定义](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/main/src/common/sdk/nvidia/inc/ctrl/ctrl0000/ctrl0000gpu.h) 定义了这些独立的编号和查询结构。

```c
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <string.h>
#include "nvos.h"
#include "class/cl0080.h"
#include "ctrl/ctrl0000/ctrl0000gpu.h"

#define RM_ALLOC_CMD(t)   _IOWR('F', 0x2B, t)
#define RM_CONTROL_CMD(t) _IOWR('F', 0x2A, t)

#define NV2080_CTRL_CMD_BUS_SET_PCIE_SPEED   0x20801805

typedef struct {
    NvU32 busSpeed;
} SET_SPEED_PARAMS;

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: %s <0|1|2|3> <device-minor> <PCI-bus-hex>\n", argv[0]);
        return 2;
    }
    NvU32 mbps = strtoul(argv[1], NULL, 10);
    if (strlen(argv[1]) != 1 || argv[1][0] < '0' || argv[1][0] > '3') return 2;
    int minor = argc > 2 ? atoi(argv[2]) : 0;

    char devpath[32];
    snprintf(devpath, sizeof(devpath), "/dev/nvidia%d", minor);
    int gfd = open(devpath, O_RDWR);
    if (gfd < 0) { perror(devpath); return 1; }

    int fd = open("/dev/nvidiactl", O_RDWR);
    if (fd < 0) { perror("open /dev/nvidiactl"); return 1; }

    struct { int rc; unsigned status; } completion = {0};
    if (ioctl(gfd, _IOWR('F', 218, typeof(completion)), &completion) < 0 || completion.rc || completion.status) {
        fprintf(stderr,"GPU open failed rc=%d status=0x%x\n",completion.rc,completion.status); return 1;
    }
    int control_fd = fd;
    if (ioctl(gfd, _IOWR('F', 201, int), &control_fd) < 0) {
        perror("register control fd"); return 1;
    }

    NVOS21_PARAMETERS p21;

    /* root client (RM generates the handle) */
    memset(&p21, 0, sizeof(p21));
    p21.hRoot = 0; p21.hObjectParent = 0; p21.hObjectNew = 0; p21.hClass = 0;
    if (ioctl(fd, RM_ALLOC_CMD(NVOS21_PARAMETERS), &p21) || p21.status) {
        fprintf(stderr, "root alloc failed status=0x%x\n", p21.status);
        return 1;
    }
    NvHandle hClient = p21.hObjectNew;

    NV0000_CTRL_GPU_GET_ATTACHED_IDS_PARAMS ids;
    memset(&ids, 0xff, sizeof(ids));
    NVOS54_PARAMETERS query = {0};
    query.hClient=hClient; query.hObject=hClient;
    query.cmd=NV0000_CTRL_CMD_GPU_GET_ATTACHED_IDS;
    query.params=&ids; query.paramsSize=sizeof(ids);
    if (ioctl(fd,RM_CONTROL_CMD(NVOS54_PARAMETERS),&query) || query.status) return 1;
    unsigned targetBus=strtoul(argv[3],NULL,16), instance=~0U;
    for(unsigned i=0;i<32 && ids.gpuIds[i]!=~0U;i++) {
        NV0000_CTRL_GPU_GET_PCI_INFO_PARAMS pci={0}; pci.gpuId=ids.gpuIds[i];
        query.cmd=NV0000_CTRL_CMD_GPU_GET_PCI_INFO; query.params=&pci; query.paramsSize=sizeof(pci); query.status=0;
        if(ioctl(fd,RM_CONTROL_CMD(NVOS54_PARAMETERS),&query) || query.status) return 1;
        if(pci.domain || pci.bus!=targetBus || pci.slot) continue;
        NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS info={0}; info.gpuId=pci.gpuId;
        query.cmd=NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2; query.params=&info; query.paramsSize=sizeof(info); query.status=0;
        if(ioctl(fd,RM_CONTROL_CMD(NVOS54_PARAMETERS),&query) || query.status) return 1;
        if (info.subDeviceInstance != 0) {
            fprintf(stderr,"Nonzero subdevice instance is unsupported\n"); return 1;
        }
        instance=info.deviceInstance;
        printf("bus=%02x gpuId=%x deviceInstance=%u gpuInstance=%u minor=%d\n",pci.bus,pci.gpuId,instance,info.gpuInstance,minor);
        break;
    }
    if(instance==~0U) { fprintf(stderr,"GPU bus not found\n"); return 1; }
    if(mbps==0) return 0;
    /* NV01_DEVICE_0, gpuInstance 0 */
    NV0080_ALLOC_PARAMETERS devParms;
    memset(&devParms, 0, sizeof(devParms));
    devParms.deviceId = instance;
    memset(&p21, 0, sizeof(p21));
    p21.hRoot = hClient; p21.hObjectParent = hClient;
    p21.hObjectNew = 0; p21.hClass = 0x80;      /* NV01_DEVICE_0 */
    p21.pAllocParms = &devParms; p21.paramsSize = sizeof(devParms);
    if (ioctl(fd, RM_ALLOC_CMD(NVOS21_PARAMETERS), &p21) || p21.status) {
        fprintf(stderr, "device alloc failed status=0x%x\n", p21.status);
        return 1;
    }
    NvHandle hDevice = p21.hObjectNew;

    /* NV20_SUBDEVICE_0 */
    memset(&p21, 0, sizeof(p21));
    p21.hRoot = hClient; p21.hObjectParent = hDevice;
    p21.hObjectNew = 0; p21.hClass = 0x2080;
    if (ioctl(fd, RM_ALLOC_CMD(NVOS21_PARAMETERS), &p21) || p21.status) {
        fprintf(stderr, "subdevice alloc failed status=0x%x\n", p21.status);
        return 1;
    }
    NvHandle hSubdevice = p21.hObjectNew;

    /* SET_PCIE_SPEED: paramsSize MUST be exactly 4 */
    SET_SPEED_PARAMS sp = { mbps };
    NVOS54_PARAMETERS p54;
    memset(&p54, 0, sizeof(p54));
    p54.hClient    = hClient;
    p54.hObject    = hSubdevice;
    p54.cmd        = NV2080_CTRL_CMD_BUS_SET_PCIE_SPEED;
    p54.params     = &sp;
    p54.paramsSize = sizeof(sp);

    if (ioctl(fd, RM_CONTROL_CMD(NVOS54_PARAMETERS), &p54) != 0 || p54.status != 0) {
        fprintf(stderr, "SET_PCIE_SPEED failed status=0x%x\n", p54.status);
        return 1;
    }
    printf("SET_PCIE_SPEED(%u) OK\n", mbps);
    return 0;
}

```

在匹配版本的 NVIDIA 开源模块源码根目录编译，不需要阅读头文件，也不需要安装 CUDA Toolkit：

```bash
# 将上面的文件放在当前目录
cc -std=gnu11 -O2 -Wall -Wextra \
  -Isrc/common/sdk/nvidia/inc \
  set_pcie_speed.c -o set_pcie_speed
sudo install -m 0755 set_pcie_speed /usr/local/sbin/cmp-pcie-speed
```

# 完整训练流程

保存以下内容为 `cmp-gen3.sh`。脚本只对明确指定的一张卡执行一次训练，不进行无限重试。已经是 Gen3 时直接验证并退出。若任何 RM 调用失败，立即停止。

```bash
#!/usr/bin/env bash
set -euo pipefail
[[ $EUID == 0 ]] || { echo '需要 root'; exit 1; }
BDF=${1:?用法: sudo bash cmp-gen3.sh 0000:03:00.0}
[[ $BDF =~ ^0000:[0-9a-fA-F]{2}:00\.0$ ]] || {
  echo '此工具只支持 0000:xx:00.0'; exit 1;
}
BDF=${BDF,,}
D=/sys/bus/pci/devices/$BDF
[[ $(cat "$D/vendor") == 0x10de && $(cat "$D/device") == 0x2082 ]] || {
  echo '不是本文验证的 10de:2082'; exit 1;
}
PARENT=$(basename "$(dirname "$(readlink -f "$D")")")
[[ $PARENT =~ ^[[:xdigit:]]{4}:[[:xdigit:]]{2}:[[:xdigit:]]{2}\.[0-7]$ ]] || exit 1
[[ $(cat "/sys/bus/pci/devices/$PARENT/class") == 0x0604* ]] || {
  echo '上游设备不是 PCI 桥'; exit 1;
}
MINOR=$(awk -F: '/Device Minor/ {gsub(/[[:space:]]/, "", $2); print $2}' "/proc/driver/nvidia/gpus/$BDF/information")
[[ $MINOR =~ ^[0-9]+$ ]] || { echo '找不到 Device Minor'; exit 1; }
BUS=${BDF:5:2}
RM=/usr/local/sbin/cmp-pcie-speed
[[ -x $RM ]] || { echo "缺少 $RM"; exit 1; }
read_status() {
  local s
  s=$(setpci -s "$1" CAP_EXP+12.w)
  [[ $s =~ ^[[:xdigit:]]{4}$ && ${s,,} != ffff ]] || return 1
  printf '%s' "$s"
}
BEFORE=$(read_status "$BDF")
WIDTH=$(( (16#$BEFORE >> 4) & 63 ))
(( WIDTH > 0 )) || { echo '链路宽度无效'; exit 1; }
echo "GPU=$BDF upstream=$PARENT minor=$MINOR before=$BEFORE"
"$RM" 0 "$MINOR" "$BUS"
if (( (16#$BEFORE & 15) != 3 )); then
  "$RM" 1 "$MINOR" "$BUS"
  sleep 1
  "$RM" 2 "$MINOR" "$BUS"
  sleep 1
  setpci -s "$PARENT" CAP_EXP+30.w=0003:000f
  setpci -s "$BDF" CAP_EXP+30.w=0003:000f
  "$RM" 3 "$MINOR" "$BUS"
  sleep 1
fi
AFTER=$(read_status "$BDF")
UP=$(read_status "$PARENT")
GEN=$((16#$AFTER & 15))
NEW_WIDTH=$(( (16#$AFTER >> 4) & 63 ))
echo "after=$AFTER upstream=$UP => Gen$GEN x$NEW_WIDTH"
(( GEN == 3 && (16#$UP & 15) == 3 && NEW_WIDTH >= WIDTH )) || {
  echo '未通过：没有建立 Gen3，或训练后宽度下降'; exit 1;
}
lspci -s "$BDF" -vv | grep -E 'LnkCap:|LnkCtl2:|LnkSta:'
echo 'Gen3 链路已建立；仍需进行负载和重启验证。'
```

运行：

```bash
sudo bash cmp-gen3.sh 0000:03:00.0
```

PCIe 配置空间的 `CAP_EXP+30.w` 是 Link Control 2，低四位 `3` 表示目标 8 GT/s。上面的掩码写入只修改目标速率字段；实际训练由 RM 完成。这里的 `3` 与驱动内部寄存器的速率编码不是同一套枚举，不能互相替换。

# 如何验收

```bash
BDF=0000:03:00.0
sudo lspci -s "$BDF" -vv | grep -E 'LnkCap:|LnkCtl2:|LnkSta:'
nvidia-smi --query-gpu=pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current --format=csv
sudo journalctl -k -b | grep -Ei 'NVRM|Xid|AER|PCIe Bus Error'
```

成功关键输出是 `LnkSta: Speed 8GT/s, Width xN`。`LnkCap` 表示能力，`LnkCtl2` 表示目标，均不能代替 `LnkSta`。Gen3 与链路宽度相互独立：只有 x4 布线的插槽建立 Gen3 x4 也是 Gen3，本文不会把它变成 x16。

运行原本已能工作的计算程序和主机与显卡数据传输测试，在负载期间再次检查速率、宽度与日志。测试应覆盖实际工作持续时间，且没有新出现的 Xid、AER 错误或宽度下降。跑分软件推断的“x16”等标签不能替代 PCIe 实际状态。

重启后再次检查。只有重启后也能正常初始化和训练，并且负载通过，才考虑开机自动运行。自动化必须在 NVIDIA 初始化完成后、计算任务启动前执行同一个单次脚本；失败应退出并留日志，不要自动反复训练或自动重启。

# 故障处理与恢复

| 现象 | 检查及处理 |
| --- | --- |
| 目标 8 GT/s，实际仍为 5 GT/s | 检查 GSP 启动后的 Gen3 代码是否实际运行、fuse 覆盖是否生效、每设备参数是否覆盖全局值，以及上游端口能力。不要只反复执行 setpci。 |
| RM 返回 0x1b | 本次排障出现过设备 minor 与 RM deviceInstance 混用导致的失败。先核对映射，再检查权限；sudo 本身不能修复错误映射。 |
| RM 返回 0x0f 或 GPU has fallen off the bus | 停止所有写入和重试，保留内核日志。正常重启恢复；重启无效时可能需要完整断电再上电。一次失败不能单独确定是显卡或主板硬件故障。 |
| Gen3 建立但宽度下降 | 不算按原宽度成功。检查插槽、转接、接触及上游链路状态，恢复后再诊断。 |
| 重启后恢复 Gen1 或 Gen2 | 检查配置是否进入实际启动路径、旧模块是否仍在 initramfs，以及训练是否需在驱动初始化后再次执行。 |
| OpenCL 仍无法运行 | Gen3 建链与 OpenCL 驱动栈健康是不同验收项；检查用户态库、ICD、设备初始化及原有 SM/TPC 配置。 |

GPU 仍然可访问时，可以请求回到 Gen2：

```bash
# minor 和 BUS 必须按上面同一张卡的实际值填写
sudo /usr/local/sbin/cmp-pcie-speed 2 "$MINOR" "$BUS"
```

若需要持久回退，应恢复操作前备份的驱动模块和配置，按原方式更新启动镜像后重启。不能把“仅将 CmpGen3 改为 0”当作对所有内部寄存器的运行时撤销。保留已经稳定的 fuse 和 SM/TPC 配置，不要借回退机会套用其他卡的屏蔽结果。

# 可复现记录

每次保存 GPU 型号及 BDF、主板插槽、内核和驱动版本、驱动源码提交或补丁副本、最终模块参数、训练前后两端 `lspci -vv`、RM 输出和内核日志。这样才能区分“能力已开放”“本次已建链”“长期稳定”三种不同结果。
