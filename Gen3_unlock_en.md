# CMP 170HX PCIe Gen3 Enablement Guide

Enabling PCIe Gen3 on a CMP 170HX requires three conditions: a software fuse override that permits Gen3 during startup, the GPU's internal Gen3 configuration, and successful link training by the driver and upstream port. Setting PCIe Target Link Speed, or seeing Gen3 listed as the maximum capability, does not prove that a Gen3 link has been established.

This guide applies to Linux x86_64 and GA100 CMP 170HX cards (the device tested was `10de:2082`), with working NVIDIA open kernel modules and an existing fuse override implementation. The reference driver version is **615.71.09**. All procedure-specific code is included here; you do not need to read the author's separate scripts. Building still requires NVIDIA source matching the driver and the system's build tools.

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
