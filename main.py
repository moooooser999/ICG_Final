from utils import *

# 立即對目前圖片執行
test_image = Image.open("./test.jpg").convert("RGB")
# plot_all_templates_for_image(test_image, rotation=30)
template_sectors = get_harmonic_templates(rotation=30)
arr = np.array(test_image.convert("RGB"))
hues = []
saturations = []

# 提取 hue
for r, g, b in arr.reshape(-1, 3):
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    hues.append(h * 360)
    saturations.append(s)
hues_deg = np.array(hues)
saturations = np.array(saturations)

min_alpha = 100
best_sector_type = None
for template in template_sectors.keys():
    best_alpha_x, best_f_x = find_optimal_alpha_minimize_scalar(
        hues_deg, saturations, template_type=template
    )
    print(f"Template: {template}, α: {best_alpha_x}, f: {best_f_x}")
    name, template_sector = get_harmonic_template_by_type(
        rotation=best_alpha_x, type_=template
    )
    # draw_hue_histogram_with_template(
    #     test_image,
    #     template_sector,
    #     title=f"Hue Histogram + {name}-type Template Best α: {best_alpha_x}",
    # )
    # if best_alpha_x < min_alpha:
    #     min_alpha = best_alpha_x
    #     best_template = template
    #     best_sector = template_sector
    #     best_sector_type = name
    naive_shifted_image = naive_hue_shifting(test_image, template_sector)
    # naive_shifted_image.save(f"naive_shifted_{name}.jpg")
# save_image(naive_shifted_image, "naive_shifted.jpg")
# naive_shifted_image.show()
