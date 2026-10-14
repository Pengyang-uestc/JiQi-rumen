"""使用手写卷积实测valid/same；Pillow仅用于读写。"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy_convolution import conv2d


def main():
    root = Path(__file__).resolve().parent
    out = root / 'padding_results'
    out.mkdir(exist_ok=True)
    with Image.open(root / 'image/img.png') as im:
        image = np.array(im.convert('RGB'))
    h, w = image.shape[:2]
    raw, rows = {}, []
    for size in (3, 5):
        kernel = np.ones((size, size)) / size**2
        for stride in (1, 2):
            for padding in ('valid', 'same'):
                name = f'k{size}_s{stride}_{padding}'
                print(name, flush=True)
                result = conv2d(image, kernel, padding=padding, stride=stride)
                expected = ((h-size)//stride+1, (w-size)//stride+1) if padding == 'valid' else ((h+stride-1)//stride, (w+stride-1)//stride)
                assert result.shape[:2] == expected
                raw[name] = result
                rows.append(dict(kernel=size, stride=stride, padding=padding, shape=list(result.shape),
                                 top_left_rgb=result[0,0].tolist(), minimum=float(result.min()), maximum=float(result.max())))
                Image.fromarray(np.rint(np.clip(result,0,255)).astype(np.uint8)).save(out / f'{name}.png')
        radius = size // 2
        # 同一原图位置比较：valid左上输出对应原图(radius,radius)。
        np.testing.assert_allclose(raw[f'k{size}_s1_valid'],
                                   raw[f'k{size}_s1_same'][radius:-radius,radius:-radius], atol=1e-10)
    # 用反射填充说明：same仅决定尺寸，边缘如何补值是另一个选择。
    reflected = conv2d(np.pad(image, ((1,1),(1,1),(0,0)), mode='reflect'),
                       np.ones((3,3))/9, padding='valid')
    raw['k3_s1_reflect'] = reflected
    fig, axes = plt.subplots(2,4, figsize=(16,7))
    panels = [image, raw['k3_s1_valid'], raw['k3_s1_same'], reflected]
    titles = ['Original', '3x3 valid', '3x3 same (zero)', '3x3 same (reflect)']
    for col,(data,title) in enumerate(zip(panels,titles)):
        display = np.rint(np.clip(data,0,255)).astype(np.uint8)
        axes[0,col].imshow(display)
        axes[0,col].set_title(f'{title}\n{data.shape}')
        axes[1,col].imshow(display[:20,:60], interpolation='nearest')
        axes[1,col].set_title('Top-left 20 rows x 60 columns')
        axes[0,col].axis('off')
        axes[1,col].axis('off')
    fig.tight_layout()
    fig.savefig(out/'comparison.png',dpi=160)
    plt.close(fig)
    np.savez_compressed(out/'raw_results.npz',**raw)
    report = dict(input_shape=list(image.shape), original_top_left=image[0,0].tolist(),
                  reflect_top_left=reflected[0,0].tolist(), experiments=rows,
                  interior_alignment_verified=True)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
