from PIL import Image, ImageDraw, ImageFont
import random

def handwritten_name_art(name, save_path="handwritten_name.png"):
    """手写体艺术签名（模拟钢笔手写）"""
    # 1. 创建画布（米色背景更贴近纸张）
    width, height = 600, 200
    img = Image.new("RGB", (width, height), (255, 248, 230))  # 米黄色背景
    draw = ImageDraw.Draw(img)

    # 2. 加载手写字体（推荐替换为本地手写字体，如「方正字迹-行书.ttf」）
    try:
        # 优先加载自定义手写字体（可替换为自己的字体路径）
        font = ImageFont.truetype("C:/Windows/Fonts/STXINGKA.TTF", 100)  # 华文行楷
    except:
        font = ImageFont.load_default(size=100)  # 备用字体

    # 3. 模拟手写抖动+笔锋（核心创意）
    name_list = list(name)
    x, y = 50, 100  # 起始位置
    pen_width = 3    # 笔锋宽度

    for i, char in enumerate(name_list):
        # 手写抖动：随机偏移（模拟手写不规整）
        dx = random.randint(-2, 2)
        dy = random.randint(-2, 2)
        # 笔锋颜色：轻微渐变（模拟墨水浓淡）
        r = random.randint(10, 30)
        g = random.randint(10, 30)
        b = random.randint(10, 30)
        # 绘制字符（带轻微旋转）
        draw.text(
            (x + dx, y + dy),
            char,
            font=font,
            fill=(r, g, b),
            stroke_width=1,  # 描边增强笔锋
            stroke_fill=(50, 50, 50)
        )
        # 连笔效果：字符间距随机（模拟手写连笔）
        char_width = draw.textlength(char, font=font)
        x += char_width + random.randint(5, 15)

    # 4. 添加纸张纹理（轻微杂色）
    for _ in range(1000):
        px = random.randint(0, width-1)
        py = random.randint(0, height-1)
        img.putpixel((px, py), (random.randint(240, 255), random.randint(230, 245), random.randint(220, 235)))

    # 5. 保存+展示
    img.save(save_path)
    img.show()
    print(f"手写体艺术签名已保存：{save_path}")

# 调用示例（替换名字）
handwritten_name_art("何也吉吉")