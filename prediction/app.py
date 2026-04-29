import os
import torch
import gradio as gr
import numpy as np
from cls import ASCClassifier   # 确保 cls.py 在同一目录下

# ------------------------------
# 配置路径（请根据实际情况调整）
# ------------------------------
MODEL_DIR = "./models"                # 存放 .pt 文件的文件夹
SPECIES_TXT = "./best_species_kfold.txt"   # 五个菌种列表文件
DEVICE = torch.device("cpu" if torch.cuda.is_available() else "cpu")

# ------------------------------
# 1. 加载菌种列表
# ------------------------------
with open(SPECIES_TXT, 'r') as f:
    species_list = [line.strip() for line in f if line.strip()]
if len(species_list) != 5:
    print(f"警告：菌种列表长度 {len(species_list)}，预期 5 个，将使用前5个（或补全）")
    if len(species_list) > 5:
        species_list = species_list[:5]
    elif len(species_list) < 5:
        # 不足5个时用占位符填充（实际不应发生）
        species_list += [f"species_{i+1}" for i in range(5 - len(species_list))]
print(f"关键菌种列表：{species_list}")

# ------------------------------
# 2. 加载所有模型
# ------------------------------
model_files = [os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) if f.endswith('.pt')]
model_files.sort()
if not model_files:
    raise RuntimeError(f"在 {MODEL_DIR} 中未找到任何 .pt 模型文件")
print(f"找到 {len(model_files)} 个模型文件")

models = []
model_input_dim = None
transform_clr = True
clr_eps = 1e-6

# 先从第一个模型读取公共参数和输入维度
first_ckpt = torch.load(model_files[0], map_location='cpu')
train_args0 = first_ckpt.get('args', {})
transform_clr = train_args0.get('transform_clr', True)
clr_eps = train_args0.get('clr_eps', 1e-6)

# 推断模型输入维度（应该等于菌种数量，但以防万一）
state_dict0 = first_ckpt['model_state_dict']
for key in state_dict0.keys():
    if 'gut_encoder.encoder.0.weight' in key:
        model_input_dim = state_dict0[key].shape[1]
        break
if model_input_dim is None:
    raise ValueError("无法从模型权重推断输入维度")
print(f"模型期望输入维度：{model_input_dim}")

# 逐个加载模型
for path in model_files:
    ckpt = torch.load(path, map_location='cpu')
    state_dict = ckpt['model_state_dict']
    train_args = ckpt.get('args', {})

    gut_latent_dim = ckpt.get('gut_dim', train_args.get('gut_latent_dim', 128))
    metadata_latent_dim = train_args.get('metadata_latent_dim', 64)
    classifier_hidden_dims = train_args.get('classifier_hidden_dims', [128, 64, 32])
    dropout = train_args.get('dropout', 0.3)
    use_batch_norm = train_args.get('use_batch_norm', True)
    use_metadata = train_args.get('use_metadata', True)
    metadata_dim = train_args.get('metadata_dim', 0)

    model = ASCClassifier(
        gut_dim=model_input_dim,
        metadata_dim=metadata_dim,
        gut_latent_dim=gut_latent_dim,
        metadata_latent_dim=metadata_latent_dim,
        classifier_hidden_dims=classifier_hidden_dims,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
        use_metadata=use_metadata,
    ).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    models.append(model)
print(f"成功加载 {len(models)} 个模型")

# ------------------------------
# 3. 预测函数（接收5个丰度值）
# ------------------------------
def predict(abundance1, abundance2, abundance3, abundance4, abundance5):
    """
    输入五个菌种的相对丰度，返回 ASD 预测分数（0~1之间）
    """
    abundances = np.array([abundance1, abundance2, abundance3, abundance4, abundance5], dtype=np.float32)

    # 构造模型输入向量（维度 = model_input_dim）
    raw = np.zeros(model_input_dim, dtype=np.float32)
    # 将用户输入的丰度填入前5个位置（如果模型维度大于5，后面补0；若小于5，则截断）
    n_user = min(len(abundances), model_input_dim)
    raw[:n_user] = abundances[:n_user]

    # 应用 arcsinh 变换（与训练时一致）
    if transform_clr:
        X = np.arcsinh(raw / clr_eps)
    else:
        X = raw

    X_tensor = torch.from_numpy(X.reshape(1, -1)).to(DEVICE)

    # 集成预测
    all_probs = []
    with torch.no_grad():
        for model in models:
            # 获取模型的元数据维度（训练时可能使用了元数据，此处用零填充）
            metadata_dim = getattr(model, 'metadata_dim', 0)
            if metadata_dim > 0:
                x_metadata = torch.zeros(1, metadata_dim, device=DEVICE)
            else:
                x_metadata = torch.zeros(1, 0, device=DEVICE)

            outputs = model(X_tensor, x_metadata)
            prob = outputs["probabilities"].cpu().item()
            all_probs.append(prob)

    avg_prob = np.mean(all_probs).item()
    return avg_prob

# ------------------------------
# 4. 构建 Gradio 界面
# ------------------------------
# 为每个菌种创建一个数字输入框
inputs = [
    gr.Number(label=species_list[0], value=0.0),
    gr.Number(label=species_list[1], value=0.0),
    gr.Number(label=species_list[2], value=0.0),
    gr.Number(label=species_list[3], value=0.0),
    gr.Number(label=species_list[4], value=0.0),
]
outputs = gr.Number(label="ASD 预测分数", precision=4)

demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=outputs,
    title="ASD 菌群风险预测",
    description="请输入以下五个关键菌种的相对丰度（取值范围建议 0~1，实际可按您的数据范围输入）",
    article="预测分数越接近 1，表示 ASD 风险越高。本工具基于集成模型（多模型平均）给出评估结果。",
    # allow_flagging="never",   # 注释掉或删除这一行
)

# ------------------------------
# 5. 启动应用
# ------------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)