# Agent Notes

## Python Environment

本项目使用 Conda 环境 `eai-swarm`（或 4060 机器上的等价环境）。

```bash
conda run -n eai-swarm python -m unittest discover -s tests
```

## Documents

所有文档用中文书写。

## Storage Boundary

- 代码、文档、脚本放在仓库。
- 训练 checkpoint、metrics、PX4 运行日志等数据放在移动硬盘
  `/Volumes/Expansion/...`，不提交到 git。

## Core Principle

`Assume AI can fail`。LLM 与 MARL 都视为不可信组件，Runtime Assurance 保留最终
硬安全权。任何新消息格式必须先冻结到 `swarm/interfaces.py` 并加测试。
