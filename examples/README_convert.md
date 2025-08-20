# Demo Conversion Tool

这个工具用于将通过 `record_demos.py` 录制的demo数据转换为可供 `record_demos_octo.py` 项目使用的格式。

## 主要差异

原始的 `record_demos.py` 和 `record_demos_octo.py` 之间的主要差异：

1. **数据处理步骤**：
   - `record_demos_octo.py` 会对成功的轨迹添加额外处理：
     - Monte Carlo returns (蒙特卡洛回报)
     - Embeddings (嵌入向量)
     - Next embeddings (下一状态嵌入向量)

2. **模型依赖**：
   - 需要加载 Octo 模型来生成嵌入向量

3. **参数设置**：
   - 添加了 `reward_scale` 和 `reward_bias` 参数

## 使用方法

### 方法1：直接转换单个文件

```bash
python convert_demos.py \
    --input_file=/path/to/your/demo.pkl \
    --exp_name=your_experiment_name \
    --reward_scale=1.0 \
    --reward_bias=0.0
```

### 方法2：使用辅助脚本（推荐）

```bash
# 交互式选择文件
python convert_demos_helper.py --exp_name=your_experiment_name

# 批量转换目录中的所有demo文件
python convert_demos_helper.py --exp_name=your_experiment_name --batch
```

## 参数说明

- `--input_file`: 输入的demo文件路径（.pkl格式）
- `--exp_name`: 实验名称，必须在 CONFIG_MAPPING 中存在
- `--reward_scale`: 奖励缩放因子（默认: 1.0）
- `--reward_bias`: 奖励偏置（默认: 0.0）
- `--discount`: 折扣因子（默认: 0.99）
- `--output_dir`: 输出目录（默认: ./demo_data）

## 输出

转换后的文件将保存在指定的输出目录中，文件名格式为：
```
{原文件名}_converted_octo_{时间戳}.pkl
```

## 验证

脚本会自动验证转换后的数据格式，确保包含所有必需的字段：
- 基本字段：observations, actions, next_observations, rewards, masks, dones, infos
- 新增字段：mc_returns, embeddings

## 示例

假设你有一个名为 `twist_20_demos_2024-01-01_12-00-00.pkl` 的demo文件：

```bash
python convert_demos.py \
    --input_file=./demo_data/twist_20_demos_2024-01-01_12-00-00.pkl \
    --exp_name=twist
```

转换后会生成：
```
./demo_data/twist_20_demos_2024-01-01_12-00-00_converted_octo_2024-01-01_12-30-00.pkl
```

## 注意事项

1. 确保目标项目的环境已正确设置
2. 确保 Octo 模型路径在配置中正确设置
3. 转换过程可能需要一些时间，特别是对于大型demo文件
4. 如果某个轨迹转换失败，脚本会跳过该轨迹并继续处理其他轨迹

## 故障排除

如果遇到问题：

1. **CONFIG_MAPPING 错误**：确保 `--exp_name` 在实验配置中存在
2. **模型加载错误**：检查 Octo 模型路径配置
3. **内存不足**：对于大型demo文件，可能需要增加系统内存或分批处理
4. **依赖缺失**：确保安装了所有必需的包（octo, data_util等）

## 文件结构

```
examples/
├── convert_demos.py          # 主转换脚本
├── convert_demos_helper.py   # 辅助脚本
├── record_demos.py          # 原始录制脚本
├── record_demos_octo.py     # 目标录制脚本
└── README_convert.md        # 此说明文件
```
