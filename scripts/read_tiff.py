# from PIL import Image

# # 读取原始TIFF文件
# img = Image.open("/raid/home/xukai/FRATTVAE/scripts/PTraj-Diff_rbg.png")

# # 获取当前分辨率（若无则默认返回 None）
# current_dpi = img.info.get("dpi", (72, 72))  # 默认值72x72

# # 修改DPI为300x300
# new_dpi = (300, 300)

# # 保存新文件（保留原始元数据并覆盖DPI）
# img.save(
#     "PTraj-Diff_Frame_300dpi.tif",
#     dpi=new_dpi,
#     compression="tiff_deflate"  # 可选压缩方式：无/deflate/lzw
# )

from admet_ai import ADMETModel

model = ADMETModel()
preds = model.predict(smiles="O(c1ccc(cc1)CCOC)CC(O)CNC(C)C")
print(type(preds))

# dict
# 'hERG' 'AMES' 'DILI'