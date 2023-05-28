import matplotlib
import numpy as np
from PIL import Image


def mandelbrot_set(width: int, height: int, real_range: tuple[float, float], imag_range: tuple[float, float],
                   max_iter: int):
    real = np.linspace(real_range[0], real_range[1], width)
    imag = np.linspace(imag_range[0], imag_range[1], height)
    c = np.ravel(real + imag[:, None] * 1j)

    z = np.zeros_like(c, dtype=np.complex128)
    mask = np.ones_like(c, dtype=bool)
    steps = np.zeros_like(c)

    for i in range(max_iter):
        z[mask] = z[mask] * z[mask] + c[mask]
        mask = np.logical_and(mask, np.abs(z) < 4)
        steps[mask] = i
    return np.abs(steps.reshape((height, width)))


def save_mandelbrot(filename: str, mandelbrot):
    mandelbrot_scaled = mandelbrot / np.max(mandelbrot)

    cmap = matplotlib.colormaps.get_cmap('hot')
    mandelbrot_colored = cmap(mandelbrot_scaled)

    image = Image.fromarray((mandelbrot_colored[:, :, :3] * 255).astype(np.uint8), mode='RGB')

    image.save(filename)


def zoom(width: int, height: int, real_range: tuple[float, float], imag_range: tuple[float, float], max_iter: int,
         zoom_factor: float, num_iterations: int):
    for i in range(num_iterations):
        real_range, imag_range = zoomed_ranges(real_range, imag_range, zoom_factor)
        max_iter = round(max_iter * 1.2)
        mandelbrot = mandelbrot_set(width, height, real_range, imag_range, max_iter)

        save_mandelbrot(f'mandelbrot_zoom_{i + 1}.png', mandelbrot)


def zoomed_ranges(real_range: tuple[float, float], imag_range: tuple[float, float], zoom_factor: float):
    real_center = (real_range[0] + real_range[1]) / 2
    imag_center = (imag_range[0] + imag_range[1]) / 2
    real_range = (real_center - (real_center - real_range[0]) * zoom_factor,
                  real_center + (real_range[1] - real_center) * zoom_factor)
    imag_range = (imag_center - (imag_center - imag_range[0]) * zoom_factor,
                  imag_center + (imag_range[1] - imag_center) * zoom_factor)

    return real_range, imag_range


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    width = 800
    height = 550
    real_range = (-2.6, 1.2)
    imag_range = (-1.1, 1.6)
    max_iter = 50

    zoom(width, height, real_range, imag_range, max_iter, 0.8, 50)
