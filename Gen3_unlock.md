# CMP 170HX PCIe Gen3 开启指南

CMP 170HX 的 PCIe Gen3 需要三个条件同时成立：启动阶段允许 Gen3 的 fuse 软件覆盖、GPU 内部 Gen3 配置、驱动与上游端口完成链路训练。只设置 PCIe Target Link Speed，或者只看到最大能力为 Gen3，都不能证明链路已经建立。

本文适用于 Linux x86_64、GA100 CMP 170HX（本次实际设备 ID 为 `10de:2082`）、已能正常加载的 NVIDIA 开源内核模块及已实现的 fuse 覆盖功能。参考驱动版本为 **615.71.09**。完整操作代码均列在正文中；不需要阅读作者的其他脚本。编译仍需要与驱动匹配的 NVIDIA 源码和系统编译工具。

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
