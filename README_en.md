# 170HX 10G GPU Unlocking Notes

[简体中文](./README.md)

## 1. Overview

Before proceeding, follow the instructions in [170hx-rir-tool](https://github.com/lixiangwuxian/170hx-rir-tool) to unlock the GPU's `disable_sw_override PRI+RED` fuse bit. Please note that this consumes two RIR slots on your GPU and is irreversible. Make sure you understand the associated risks before proceeding.

## 2. PCIe Unlocking

See [Gen3_unlock_en.md](./Gen3_unlock_en.md). There appear to be issues that cause only the first GPU to successfully enable Gen3 in multi-GPU configurations.
