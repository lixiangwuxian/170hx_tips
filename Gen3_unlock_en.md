# CMP 170HX PCIe Gen3 Enablement Guide

Enabling PCIe Gen3 on a CMP 170HX requires three conditions: a software fuse override that permits Gen3 during startup, the GPU's internal Gen3 configuration, and successful link training by the driver and upstream port. Setting PCIe Target Link Speed, or seeing Gen3 listed as the maximum capability, does not prove that a Gen3 link has been established.

This guide applies to Linux x86_64 and GA100 CMP 170HX cards (the device tested was `10de:2082`), with working NVIDIA open kernel modules. This guide includes source integration and validation of fuse override; existing support for normal CMP initialization is still a prerequisite. The reference driver version is **615.71.09**. All procedure-specific code is included here; you do not need to read the author's separate scripts. Building still requires NVIDIA source matching the driver and the system's build tools.

# Validation scope

In the existing troubleshooting records, one card established an **8 GT/s x16** link using the RM Gen1 → Gen2 → Gen3 sequence. A second reference card also reported Gen3 x16. A Gen3 transition on another card resulted in a lost GPU. That failure alone does not establish that the card cannot support Gen3, but it does not count as a stable pass either.

“Established” means the actual link state reported by `LnkSta`. Sustained load stability, each motherboard slot, and each driver version require separate validation. The tool reproduced here includes added input checks and has not undergone a new end-to-end hardware test on another machine.

# Before starting

Stop compute workloads on the target GPU. Changing link speed can interrupt GPU access. For remote operation, ensure that rebooting and, if necessary, a full power cycle are possible. Do not perform the transition from a desktop session using the same GPU.

BDF in the commands below is the GPU's PCI address; replace the example with your actual address. On multi-GPU systems, work on one card at a time. Do not substitute the index shown by nvidia-smi for a BDF.

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

The GPU must initialize correctly and already run a CUDA or OpenCL application. Resolve existing Xid errors, lost-GPU conditions, or BAR mapping errors first. Gen3 training does not fix those problems.

To remain consistent with the implementation that succeeded, the training tool below is restricted to **PCI domain 0000, device 00, function 0**, or `0000:xx:00.0`, on a standalone GPU without SLI. Other PCI topologies require changes to the PCI matching logic. This is a limitation of the tool, not a hardware address restriction.

# Required driver configuration

## Fuse override and module parameters

The reference implementation uses `cmp_fuse_gen3=0` during probe to clear the Gen3-disabled state, involving BAR0 offsets `0x820250` and `0x820580`. This is a volatile software override. This procedure does not burn physical fuses or flash the VBIOS. Other fuse override implementations must provide equivalent behavior, although their parameter names may differ.

The reference driver uses the following parameter combination:

```text
options nvidia NVreg_RegistryDwords="RmForceEnableGen2=1;RMPcieLinkSpeed=0x5;CmpGen3=1" cmp_fuse_gen3=0
```

`RMPcieLinkSpeed=0x5` is the bitfield used by this implementation to allow Gen3; it does not mean “generation five.” `CmpGen3` is a key read by a custom patch. Adding this key to a stock driver's configuration does not implement the register operations below.

Locate the existing configuration and back up the relevant files before editing:

```bash
sudo grep -RnsE 'NVreg_RegistryDwords|cmp_fuse|CmpGen3|RMPcieLinkSpeed' /etc/modprobe.d /usr/lib/modprobe.d 2>/dev/null
modprobe -c | grep '^options nvidia '
```

Merge these keys into the existing configuration, retaining only one effective value for each key. Preserve existing SM/TPC masks, memory settings, and BAR parameters. In particular, do not copy TPC screening results from a different card. Check `NVreg_RegistryDwordsPerDevice` for conflicting overrides for the target GPU, such as `CmpGen3=0`.

### 1. Confirm that the module actually implements fuse override

`cmp_fuse_gen3` is not a stock NVIDIA parameter. Check the module you intend to install; if the driver is loaded, also check the parameter files of the running module:

```bash
modinfo -p nvidia | grep -E '^cmp_fuse_(gen3|bdf):'
# Only when nvidia is loaded: these belong to the running module, not the candidate on disk
ls /sys/module/nvidia/parameters/cmp_fuse_{gen3,bdf}
```

If these parameters are absent, adding `cmp_fuse_gen3=0` to modprobe configuration is not sufficient. Integrate the source below, then build and install it together with the later Gen3 code. If the source already defines these parameters and `nv_cmp170hx_fuse_override()`, do not paste duplicate definitions; verify that probe actually calls the function.

### 2. Implement the probe-stage fuse override

The file is `kernel-open/nvidia/nv-pci.c`. The following is the fuse override block from the 615.71.09 source used in this troubleshooting session, with only the unrelated BAR1 size parameter declarations omitted. Place it after the existing headers and before the definition of `nv_pci_probe()`. Headers already included need not be added again.

The complete parameter table and function are retained because the known GPC/TPC and other OPT values must be copied into their corresponding `_CP` shadows before enabling `EN_SW_OVERRIDE`, rather than replacing settings that were not requested with stale shadow values. **For this guide, set only `cmp_fuse_gen3=0`; do not additionally set `cmp_fuse_gpc=0`, `cmp_fuse_tpc*=0`, or `cmp_fuse_flr=1` just to enable Gen3.** Leave other parameters at their defaults or retain their existing validated settings. Inclusion of other features in this code does not imply that all those branches were validated by this guide.

<details>
<summary>Expand: complete probe fuse override code</summary>

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

Inside `nv_pci_probe()`, add one call after BAR0 validation and successful region acquisition, immediately before the existing `NV_KZALLOC(nvl, ...)`. Preserve surrounding error handling. Do not move the call before BAR0 becomes accessible or defer it until after RM/GSP initialization:

```c
    /* BAR0 validation and request_mem_region have succeeded above. */
    nv_cmp170hx_fuse_override(pci_dev, regs_bar_index);

    NV_KZALLOC(nvl, sizeof(nv_linux_state_t));
```

In this implementation, if bit 0 of `0x00820084` disables software override, the function logs the condition and skips the operation; it does not attempt to burn fuses. The normal path seeds the known `_CP` registers, enables `EN_SW_OVERRIDE` at `0x00820040`, then clears bit 0 at `0x00820250` and `0x00820580` with immediate readback. These two Gen3 entries use direct OPT writes, not TPC masks.

The default `cmp_fuse_gen3=-1` means “leave unchanged.” A value of `0` writes zero to the Gen3-disable bits, allowing Gen3; it does not disable the override feature. Changing configuration does not modify an already loaded module: rebuild/reload or reboot as described below. Reapply and verify after GPU resets as well; this is not a permanent unlock.

### 3. Configure the target card and reload

To target one card, add its full BDF restriction to the existing parameter line, for example:

```text
options nvidia NVreg_RegistryDwords="RmForceEnableGen2=1;RMPcieLinkSpeed=0x5;CmpGen3=1" cmp_fuse_bdf=0000:03:00.0 cmp_fuse_gen3=0
```

This is only an example of the Gen3-related settings to merge. Do not replace the entire existing line and lose SM/TPC, BAR, or other settings. Use the BDF enumerated on your own host. To apply the override to multiple supported CMP 170HX cards, omit `cmp_fuse_bdf` and retain its empty-string default; the code still filters for `10de:20c2/2082`. The hardware validated in this guide is `2082`; inclusion of `20c2` in the code is not evidence that it has also been validated.

`cmp_fuse_bdf` is not an ECID matcher: update or remove a stale restriction after reseating a card. If an ECID loader manages per-card TPC masks, preserve that mechanism instead of replacing it with one common TPC mask. Preserve VFIO ownership of passthrough cards and use the host's existing guarded loading procedure. `modprobe -r` may also unload soft dependencies, while `--ignore-install` or direct `insmod` can bypass a custom loader.

Complete the post-GSP Gen3 code, build/install steps, and boot-image handling below, then reboot into the new module. Even when the override code already exists and only parameters change, reload through the existing procedure after stopping GPU clients, or reboot. Repeated live register writes are not needed to inspect the state.

### 4. Verify execution and read back the post-initialization state

After reboot, check the running module's parameters and this boot's log:

```bash
BDF=0000:03:00.0
cat /sys/module/nvidia/parameters/cmp_fuse_gen3
cat /sys/module/nvidia/parameters/cmp_fuse_bdf
sudo journalctl -k -b --no-pager | grep -E "${BDF}.*CMP fuse:|CMP fuse:.*${BDF}"
```

Expect entries for the target card such as `CMP fuse: gen3 ... write=0x0 opt=0x0 (direct)`, `CMP fuse: gen3_boot ... write=0x0 opt=0x0 (direct)`, and `EN_SW_OVERRIDE=1 (pre-RM/GSP)`. Resolve `DID NOT STICK`, mapping failures, or a disabled software override before proceeding. A parameter file containing `0` alone is not proof of a successful write. Missing logs also require checking the actual loaded module, BDF restriction, and call site.

Probe logs establish the state only at that instant. Subsequent GSP/driver initialization may change it. Once the GPU initializes normally, the following read-only check inspects the current values without writing registers or switching the link:

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

`OPT_GEN23` (`0x0082057c`) is included for diagnostics only. It is not the same register as either of the two Gen3-disable bits above. Without validating the corresponding driver timing, do not treat “clear this bit as well” as a universal patch. Adding this override during troubleshooting did not resolve the cold-boot issue and was reverted; that experiment must not be presented as a validated required step.

Even if both disable bits read back as zero, the later GSP Gen3 configuration, RM training, and `LnkSta`, load, and reboot checks are still required. **A successful fuse override is not proof of a stable Gen3 link.**

## Driver code needed when only fuse override is implemented

If the driver already implements the Gen3 operations below, do not add them a second time. Otherwise, add this logic to NVIDIA open driver source that already contains your fuse override patch. A configuration file cannot supply missing driver code to a binary module that only implements fuse override.

The reference insertion point is `kgspBootstrap_TU102` in `src/nvidia/src/kernel/gpu/gsp/arch/turing/kernel_gsp_tu102.c`, after successful normal BooterLoad completion and before the function's final error handling. The operations must occur at the correct point after GSP startup, not earlier during probe. The block below consolidates the reference implementation's Gen3 operations. Adapt it first if your driver version changes the function or initialization flow; do not insert it mechanically.

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

This block is restricted to device `2082`; do not generalize it to other NVIDIA models. It is neither a standalone userspace program nor a patch file that applies to every driver version. Preserve the existing normal CMP initialization logic. This consolidated version adds checks for successful completion of polling; it has not separately passed a driver build and hardware regression test.

## Build installation and reboot

You need development headers for the running kernel, C/C++ compilers, make, and open kernel module source that exactly matches the NVIDIA userspace libraries and GSP firmware. Work in the source tree that already contains the fuse override patch. Do not replace it with pristine upstream source and lose the fuse override functionality.

Example for direct installation from source:

```bash
# Run from the patched open-gpu-kernel-modules source root matching your driver version
make -j"$(nproc)" modules
# Continue only if the build succeeds; retain old modules or a bootable previous kernel
sudo make modules_install
sudo depmod -a
```

If the existing installation uses DKMS or distribution packages, rebuild and install through that same mechanism to avoid conflicts with package-managed files. With Secure Boot enabled, sign the new modules and establish trust using your existing process; otherwise, the system may refuse to load them after reboot.

If the initramfs contains NVIDIA modules or their configuration, update it as well. Debian/Ubuntu commonly use `sudo update-initramfs -u -k "$(uname -r)"`; systems using dracut commonly use `sudo dracut -f`. Use the method appropriate for your distribution, not both. If your existing boot design deliberately excludes NVIDIA from the initramfs, preserve that design.

Save your work and reboot. After reboot, check `/proc/driver/nvidia/version`, `nvidia-smi`, and the kernel log, and confirm that the new module is actually loaded. A new version shown by `modinfo` alone does not prove that the running module has changed.

# Complete RM training tool

Save the complete source below as `set_pcie_speed.c`. It requests speed changes through NVIDIA's RM interface so that the driver performs link training. Inputs `1/2/3` mean Gen1/Gen2/Gen3 respectively; `0` only queries the mapping. **Do not pass 2500, 5000, or 8000.**

The device minor number comes from `/proc/driver/nvidia/gpus/BDF/information`, while RM deviceInstance is obtained through a GPU ID query. They are not interchangeable. NVIDIA's [GPU control interface definitions](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/main/src/common/sdk/nvidia/inc/ctrl/ctrl0000/ctrl0000gpu.h) define these separate identifiers and query structures.

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

Compile from the root of the matching NVIDIA open kernel module source tree. You do not need to read its headers or install the CUDA Toolkit:

```bash
# Save the source above in the current directory
cc -std=gnu11 -O2 -Wall -Wextra \
  -Isrc/common/sdk/nvidia/inc \
  set_pcie_speed.c -o set_pcie_speed
sudo install -m 0755 set_pcie_speed /usr/local/sbin/cmp-pcie-speed
```

# Complete training procedure

Save the following as `cmp-gen3.sh`. The script performs one training sequence on one explicitly selected card, without indefinite retries. If the link is already Gen3, it validates the state and exits. Any failed RM call stops the script immediately.

```bash
#!/usr/bin/env bash
set -euo pipefail
[[ $EUID == 0 ]] || { echo 'Root privileges required'; exit 1; }
BDF=${1:?Usage: sudo bash cmp-gen3.sh 0000:03:00.0}
[[ $BDF =~ ^0000:[0-9a-fA-F]{2}:00\.0$ ]] || {
  echo 'This tool supports only 0000:xx:00.0'; exit 1;
}
BDF=${BDF,,}
D=/sys/bus/pci/devices/$BDF
[[ $(cat "$D/vendor") == 0x10de && $(cat "$D/device") == 0x2082 ]] || {
  echo 'Device is not the validated 10de:2082'; exit 1;
}
PARENT=$(basename "$(dirname "$(readlink -f "$D")")")
[[ $PARENT =~ ^[[:xdigit:]]{4}:[[:xdigit:]]{2}:[[:xdigit:]]{2}\.[0-7]$ ]] || exit 1
[[ $(cat "/sys/bus/pci/devices/$PARENT/class") == 0x0604* ]] || {
  echo 'Upstream device is not a PCI bridge'; exit 1;
}
MINOR=$(awk -F: '/Device Minor/ {gsub(/[[:space:]]/, "", $2); print $2}' "/proc/driver/nvidia/gpus/$BDF/information")
[[ $MINOR =~ ^[0-9]+$ ]] || { echo 'Device Minor not found'; exit 1; }
BUS=${BDF:5:2}
RM=/usr/local/sbin/cmp-pcie-speed
[[ -x $RM ]] || { echo "Missing $RM"; exit 1; }
read_status() {
  local s
  s=$(setpci -s "$1" CAP_EXP+12.w)
  [[ $s =~ ^[[:xdigit:]]{4}$ && ${s,,} != ffff ]] || return 1
  printf '%s' "$s"
}
BEFORE=$(read_status "$BDF")
WIDTH=$(( (16#$BEFORE >> 4) & 63 ))
(( WIDTH > 0 )) || { echo 'Invalid link width'; exit 1; }
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
  echo 'FAILED: Gen3 was not established, or link width decreased'; exit 1;
}
lspci -s "$BDF" -vv | grep -E 'LnkCap:|LnkCtl2:|LnkSta:'
echo 'Gen3 link established; load and reboot validation are still required.'
```

Run it with your GPU's actual BDF:

```bash
sudo bash cmp-gen3.sh 0000:03:00.0
```

In PCIe configuration space, `CAP_EXP+30.w` is Link Control 2. A value of `3` in its low four bits selects a target speed of 8 GT/s. The masked writes above change only the target speed field; RM performs the actual training. This `3` uses a different encoding from the driver's internal rate register, so the two values must not be interchanged.

# Acceptance checks

```bash
BDF=0000:03:00.0
sudo lspci -s "$BDF" -vv | grep -E 'LnkCap:|LnkCtl2:|LnkSta:'
nvidia-smi --query-gpu=pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current --format=csv
sudo journalctl -k -b | grep -Ei 'NVRM|Xid|AER|PCIe Bus Error'
```

The key success output is `LnkSta: Speed 8GT/s, Width xN`. `LnkCap` describes capability and `LnkCtl2` describes the target; neither replaces `LnkSta`. Generation and link width are independent. A Gen3 x4 link in a slot wired for x4 is still Gen3; this procedure does not turn that slot into x16.

Run a previously working compute application and host-to-device/device-to-host transfer tests. Recheck speed, width, and logs while the GPU is under load. Test for a duration representative of the actual workload, with no new Xid errors, AER errors, or link-width reduction. Labels such as “x16” inferred by benchmark software are not a substitute for the actual PCIe state.

Check again after reboot. Consider automatic startup only after initialization and training work across reboots and load testing passes. Automation must run the same single-pass script after NVIDIA initialization and before compute workloads start. On failure, exit and retain logs; do not automatically retrain repeatedly or reboot.

# Troubleshooting and recovery

| Symptom | Checks and action |
| --- | --- |
| Target is 8 GT/s but actual speed remains 5 GT/s | Check that the Gen3 code runs after GSP startup, that fuse override is effective, that per-device settings do not override global values, and that the upstream port supports the speed. Repeating setpci alone is not a solution. |
| RM returns 0x1b | Confusing the device minor with RM deviceInstance caused this failure during troubleshooting. Verify the mapping before investigating permissions; sudo alone cannot correct a wrong mapping. |
| RM returns 0x0f or reports GPU has fallen off the bus | Stop all writes and retries, and retain the kernel log. Attempt recovery with a normal reboot. If that fails, a full power cycle may be necessary. A single failure does not establish a GPU or motherboard hardware fault. |
| Gen3 is established but width decreases | This is not a pass at the original width. Check the slot, adapters, contacts, and upstream link state after recovery. |
| Link returns to Gen1 or Gen2 after reboot | Check that the configuration is used during actual startup, that an old module is not still in the initramfs, and whether training must run again after driver initialization. |
| OpenCL still fails | Gen3 link establishment and a healthy OpenCL software stack are separate acceptance criteria. Check userspace libraries, the ICD, device initialization, and existing SM/TPC settings. |

If the GPU is still accessible, you can request a return to Gen2:

```bash
# Set MINOR and BUS to the actual values for the same GPU identified above
sudo /usr/local/sbin/cmp-pcie-speed 2 "$MINOR" "$BUS"
```

For a persistent rollback, restore the driver modules and configuration backed up before the change, update the boot image through the original mechanism, and reboot. Setting `CmpGen3` to `0` alone is not a runtime undo of every internal register change. Preserve stable fuse and SM/TPC settings; do not apply another card's screening results during rollback.

# Reproducibility record

For each attempt, save the GPU model and BDF, motherboard slot, kernel and driver versions, driver source commit or patch copy, final module parameters, `lspci -vv` output from both link endpoints before and after training, RM output, and the kernel log. These records distinguish “capability enabled,” “link established in this attempt,” and “stable under sustained use.”
