import os
from tqdm import tqdm
import timeit
import multiprocessing

import meep as mp
import meep.adjoint as mpa
import numpy as np
import autograd.numpy as npa
from autograd import tensor_jacobian_product

from matplotlib import pyplot as plt
from icecream import ic


class MappingClass:
    def __init__(self, **sim_kwargs):
        self.sim_kwargs = sim_kwargs

    def __call__(self, x, eta, beta):
        return mapping(x, eta, beta, **self.sim_kwargs)


def mapping(x, eta, beta, Nx, Ny, filter_radius, design_region_width,
            design_region_height, design_region_resolution, **kwargs):
    # up-down symmetry
    x = (npa.fliplr(x.reshape(Nx, Ny)) + x.reshape(Nx, Ny))/2
    # filter
    filtered_field = mpa.conic_filter(x, filter_radius, design_region_width,
                                      design_region_height,
                                      design_region_resolution)
    projected_field = mpa.tanh_projection(filtered_field, beta, eta)
    return projected_field.flatten()


def mirror_upper_y_half(x):
    """
    prend la partie superueure x[:, half:] de x et la copie sur la partie
    inferieure x[:, :half]
    """
    half = int(x.shape[1]/2)
    upper_half = x[:, half:]
    if x.shape[1] % 2 == 0:
        out = np.concatenate([np.fliplr(upper_half), upper_half], axis=1)
    if x.shape[1] % 2 == 1:
        # si ou a x est de taille impaire en y, on ne repete pas la ligne du
        # milieu
        out = np.concatenate(
            [np.fliplr(upper_half)[:, :-1], upper_half], axis=1)
    return out


def double_with_mirror(image):
    channels = '~/scratch/nanophoto/lowfom/nodata/fields/channels.npy'
    channels = np.load(os.path.expanduser(channels))
    mirrored_image = np.fliplr(image)  # Crée l'image miroir
    doubled_image = np.concatenate((mirrored_image[:, :-1], image), axis=1)
    return doubled_image


def normalise(image):
    image = (image - image.min()) / (image.max() - image.min())
    return image


def compute_FOM_and_gradient_parallele(images):
    if images.ndim == 2:
        return compute_FOM_and_gradient(images)
    images = [images[i] for i in range(images.shape[0])]
    ic(multiprocessing.cpu_count())
    with multiprocessing.Pool() as pool:
        foms, gradients = pool.map(compute_FOM_and_gradient, images)
    foms = np.array(foms)
    gradients = np.concatenate(gradients, axis=0)
    # TODO: implémnter un catch de plantage de Meep
    return foms


def compute_FOM_and_gradient(image):
    opt, sim_args = get_opt()
    Nx = sim_args['Nx']
    Ny = sim_args['Ny']

    image = double_with_mirror(image)
    image = normalise(image)

    mapping = MappingClass(**sim_args)
    x = mirror_upper_y_half(image)
    f0, g0 = opt([mapping(x, 0.5, 256)])
    f0 = f0/2
    backprop_gradient = tensor_jacobian_product(
        mapping, 0)(x, 0.5, 2, g0[:, 0])
    backprop_gradient = backprop_gradient.reshape(Nx, Ny)
    return f0, backprop_gradient


def compute_FOM_parallele(images, debug=False):
    """
        Input images are (101, 91) and are doubled to shape (101, 181) and
        normalized to have values in the range [0, 1] then meep simulates the
        FOM.

        inputs:
        images: np.array (batch_size, 101, 91)

        outputs:
        foms: np.array (batch_size)
    """
    if images.ndim == 2:
        return compute_FOM(images)
    if images.ndim > 3:
        images = images.squeeze()
        assert images.ndim == 3, f"trop de dimensions {images.ndim} > 3"

    images = [images[i] for i in range(images.shape[0])]
    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap(compute_FOM, images)))
    foms = np.array(results)
    # TODO: implémnter un catch de plantage de Meep
    return foms


def compute_FOM_array(images, debug=False):
    if images.ndim == 2:
        return compute_FOM(images), []
    foms = np.empty(0)
    error_idx = []
    for i, image in enumerate(images if debug is False else tqdm(images)):
        try:
            fom = compute_FOM(image)
            foms = np.append(foms, fom)
        except RuntimeError:
            error_idx.append(i)
    return foms, error_idx


# @mesurer_memoire_fonction
# mem max utilisee 11MB
def get_sim(symmetry_enable=True):
    # t0 = timeit.default_timer()
    # ## Basic environment setup
    pml_size = 1.0  # (μm)

    dx = 0.02
    opt_size_x = 101 * dx
    opt_size_y = 181 * dx
    size_x = 2.6 + pml_size  # um
    size_y = 4.5 + pml_size  # um
    # size_x = opt_size_x + 2*0.4
    # size_y = opt_size_y + 2*0.4
    out_wg_dist = 1.25
    wg_width = 0.5
    mode_width = 3*wg_width
    wg_index = 2.8
    bg_index = 1.44
    # wg_zspan = 0.22

    # opt_xpixel = opt_size_x*(1/dx)
    # opt_ypixel = opt_size_y*(1/dx)

    # source_wg_xmin = -size_x
    # source_wg_xmax = -opt_size_x/2 + 0.1
    # source_wg_y = 0
    # source_wg_yspan = wg_width
    # source_wg_z = 0
    # source_wg_zspan = wg_zspan

    # top_wg_xmin = opt_size_x/2 - 0.1
    # top_wg_xmax = size_x
    # top_wg_y = out_wg_dist
    # top_wg_yspan = wg_width
    # top_wg_z = 0
    # top_wg_zspan = wg_zspan

    # bot_wg_xmin = top_wg_xmin
    # bot_wg_xmax = top_wg_xmax
    # bot_wg_y = -out_wg_dist
    # bot_wg_yspan = wg_width
    # bot_wg_z = 0
    # bot_wg_zspan = wg_zspan

    source_x = -size_x/2 - 0.1
    source_y = 0
    source_yspan = mode_width
    source_z = 0
    # source_zspan = 1
    center_wavelength = 1.550

    seed = 240
    np.random.seed(seed)
    mp.verbosity(0)
    # Effective permittivity for a Silicon waveguide with a thickness of 220nm
    Si = mp.Medium(index=wg_index)
    SiO2 = mp.Medium(index=bg_index)
    # size of a pixel (in μm) 20 nm in lumerical exp
    delta = dx
    # resolution = 20 # (pixels/μm)
    resolution = 1/delta  # pixels/μm
    waveguide_width = wg_width  # 0.5 # (μm)
    design_region_width = opt_size_x  # (μm)
    design_region_height = opt_size_y  # (μm)
    # 1.0 (μm) distance between arms center to center
    arm_separation = out_wg_dist
    # waveguide_length = source_wg_xmax - source_wg_xmin  # 0.5 (μm)

    # ## Design variable setup

    minimum_length = 0.09  # (μm)
    eta_e = 0.75
    filter_radius = mpa.get_conic_radius_from_eta_e(
        minimum_length, eta_e)  # (μm)
    eta_i = 0.5
    # eta_d = 1-eta_e
    # int(4*resolution) # (pixels/μm)
    design_region_resolution = int(resolution)
    frequencies = 1/np.linspace(1.5, 1.6, 5)  # (1/μm)

    Nx = int(design_region_resolution*design_region_width)
    Ny = int(design_region_resolution*design_region_height)

    design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny), SiO2, Si)
    size = mp.Vector3(design_region_width, design_region_height)
    volume = mp.Volume(center=mp.Vector3(), size=size)
    design_region = mpa.DesignRegion(design_variables, volume=volume)

    # ## Simulation Setup

    Sx = 2*pml_size + size_x  # cell size in X
    Sy = 2*pml_size + size_y  # cell size in Y
    cell_size = mp.Vector3(Sx, Sy)

    pml_layers = [mp.PML(pml_size)]

    fcen = 1/center_wavelength  # 1/1.55
    width = 0.2
    fwidth = width * fcen
    source_center = [source_x, source_y, source_z]

    source_size = mp.Vector3(0, source_yspan, 0)
    kpoint = mp.Vector3(1, 0, 0)

    src = mp.GaussianSource(frequency=fcen, fwidth=fwidth)
    source = [mp.EigenModeSource(src,
                                 eig_band=1,
                                 direction=mp.NO_DIRECTION,
                                 eig_kpoint=kpoint,
                                 size=source_size,
                                 center=source_center,
                                 eig_parity=mp.EVEN_Z+mp.ODD_Y)]
    mon_pt = mp.Vector3(*source_center)

    sim_args = {'Nx': Nx,
                'Ny': Ny,
                'Sx': Sx,
                'Sy': Sy,
                'eta_i': eta_i,
                'filter_radius': filter_radius,
                'design_region_width': design_region_width,
                'design_region_height': design_region_height,
                'design_region_resolution': design_region_resolution,
                'fcen': fcen,
                'frequencies': frequencies,
                'source_x': source_x,
                'source_center': source_center,
                'size_x': size_x,
                'arm_separation': arm_separation,
                'mon_pt': mon_pt,
                'design_region_size': size,
                'waveguide_width': waveguide_width,
                }

    geometry = [
        # left waveguide
        mp.Block(center=mp.Vector3(x=-Sx/4),
                 material=Si,
                 size=mp.Vector3(Sx/2+1, waveguide_width, 0)),
        # top right waveguide
        mp.Block(center=mp.Vector3(x=Sx/4, y=arm_separation),
                 material=Si,
                 size=mp.Vector3(Sx/2+1, waveguide_width, 0)),
        # bottom right waveguide
        mp.Block(center=mp.Vector3(x=Sx/4, y=-arm_separation),
                 material=Si,
                 size=mp.Vector3(Sx/2+1, waveguide_width, 0)),
        mp.Block(center=design_region.center,
                 size=design_region.size,
                 material=design_variables)
    ]

    symmetries = [mp.Mirror(direction=mp.Y, phase=-1)
                  ] if symmetry_enable is True else None
    sim = mp.Simulation(cell_size=cell_size,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=source,
                        symmetries=symmetries,
                        default_material=SiO2,
                        resolution=resolution)
    return sim, design_region, sim_args


def get_opt():
    sim, design_region, sim_args = get_sim()
    waveguide_width = sim_args['waveguide_width']
    source_x = sim_args['source_x']
    arm_separation = sim_args['arm_separation']
    size_x = sim_args['size_x']
    frequencies = sim_args['frequencies']
    filter_radius = sim_args['filter_radius']
    design_region_width = sim_args['design_region_width']
    design_region_height = sim_args['design_region_height']
    design_region_resolution = sim_args['design_region_resolution']
    eta_i = sim_args['eta_i']
    Sx = sim_args['Sx']
    Sy = sim_args['Sy']
    Nx = sim_args['Nx']
    Ny = sim_args['Ny']

    size = mp.Vector3(Sx, Sy, 0)
    monsize = mp.Vector3(y=3*waveguide_width)
    source_mon_center = mp.Vector3(x=source_x + 0.1)
    top_mon_center = mp.Vector3(size_x/2, arm_separation, 0)
    # source_fluxregion = mp.FluxRegion(center=source_mon_center,
    #                                   size=monsize,
    #                                   weight=-1)
    # top_fluxregion = mp.FluxRegion(center=top_mon_center,
    #                                size=monsize,
    #                                weight=-1)

    # abs_src_coeff = 57.97435797757672

    # Get top output flux coefficients
    topmoncenter = mp.Vector3(size_x/2, arm_separation, 0)
    # topfluxregion = mp.FluxRegion(topmoncenter, monsize)

    mode = 1

    volume = mp.Volume(center=topmoncenter, size=monsize)
    ob_list = [mpa.EigenmodeCoefficient(sim, volume, mode)]

    # -------
    monsize = monsize = mp.Vector3(y=3*waveguide_width)
    source_mon_center = mp.Vector3(x=source_x + 0.1)
    TE0 = mpa.EigenmodeCoefficient(sim,
                                   mp.Volume(center=source_mon_center,
                                             size=monsize), mode)
    top_mon_center = mp.Vector3(size_x/2, arm_separation, 0)
    TE_top = mpa.EigenmodeCoefficient(sim,
                                      mp.Volume(center=top_mon_center,
                                                size=monsize), mode)

    bot_mon_center = mp.Vector3(size_x/2, -arm_separation, 0)
    TE_bottom = mpa.EigenmodeCoefficient(sim,
                                         mp.Volume(center=bot_mon_center,
                                                   size=monsize), mode)
    ob_list = [TE0, TE_top, TE_bottom]

    # def J(top):
    #     return npa.mean(npa.abs(top)**2)

    def J(source, top, bottom):
        power = npa.abs(top/source) ** 2 + npa.abs(bottom/source) ** 2
        return npa.mean(power)

    opt = mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=J,
        objective_arguments=ob_list,
        design_regions=[design_region],
        frequencies=frequencies
    )

    sim_args = {"Nx": Nx, "Ny": Ny,
                "filter_radius": filter_radius,
                "design_region_width": design_region_width,
                "design_region_height": design_region_height,
                "design_region_resolution": design_region_resolution,
                "eta_i": eta_i}
    return opt, sim_args


def compute_FOM(image, symmetry_enable=True, debug=False):
    assert image.shape == (101, 91), f"{image.shape} != (101,91)"
    assert not ((image > 1.) + (image < 0.)).any(), f""
    # t1 = timeit.default_timer()
    # ic('init sim', t1-t0)
    sim, design_region, sim_args = get_sim()

    Sx = sim_args['Sx']
    Sy = sim_args['Sy']
    fcen = sim_args['fcen']
    waveguide_width = sim_args['waveguide_width']
    source_x = sim_args['source_x']
    source_center = sim_args['source_center']
    size_x = sim_args['size_x']
    arm_separation = sim_args['arm_separation']
    mon_pt = sim_args['mon_pt']

    idx_map = double_with_mirror(image)

    index_map = mapping(idx_map, 0.5, 256, **sim_args)
    design_region.update_design_parameters(index_map)

    # full field monitor
    size = mp.Vector3(Sx, Sy, 0)
    # _ = sim.add_dft_fields(
    #     [mp.Ex, mp.Ey, mp.Ez],             # Components to monitor
    #     fcen, 0, 1,
    #     # frequency=fcen,                     # Operating frequency
    #     center=mp.Vector3(0, 0, 0),        # Center of the monitor region
    #     size=size          # Size of the monitor region
    # )

    monsize = mp.Vector3(y=3*waveguide_width)
    # source_mon_center = mp.Vector3(x=source_x + 0.1)
    # top_mon_center = mp.Vector3(size_x/2, arm_separation, 0)
    # source_fluxregion = mp.FluxRegion(center=source_mon_center,
    #                                   size=monsize,
    #                                   weight=-1)
    # top_fluxregion = mp.FluxRegion(center=top_mon_center,
    #                                size=monsize,
    #                                weight=-1)

    sim.run(until_after_sources=100)

    def get_eigenmode_coeffs(sim, fluxregion, mon_pt):
        sim.reset_meep()
        flux = sim.add_flux(fcen, 0, 1, fluxregion)
        # breakpoint()
        mon_pt = mp.Vector3(*source_center)
        sim.run(until_after_sources=mp.stop_when_fields_decayed(50,
                                                                mp.Ez,
                                                                mon_pt,
                                                                1e-9))
        # res = sim.get_eigenmode_coefficients(flux, [1])
        res = sim.get_eigenmode_coefficients(flux, [1],
                                             eig_parity=mp.EVEN_Z+mp.ODD_Y)
        coeffs = res.alpha
        return coeffs

    def get_flux_spectrum(sim, fluxregion):
        sim.reset_meep()
        flux = sim.add_flux(fcen, 0, 1, fluxregion)
        # breakpoint()
        mon_pt = mp.Vector3(*source_center)
        sim.run(until_after_sources=mp.stop_when_fields_decayed(50,
                                                                mp.Ez,
                                                                mon_pt,
                                                                1e-9))
        accumulated_flux_spectrum = mp.get_fluxes(flux)
        return accumulated_flux_spectrum

    def get_coeffs_flux_spec(sim, fluxregion):
        sim.reset_meep()
        flux = sim.add_flux(fcen, 0, 1, fluxregion)
        # breakpoint()
        mon_pt = mp.Vector3(*source_center)
        sim.run(until_after_sources=mp.stop_when_fields_decayed(50,
                                                                mp.Ez,
                                                                mon_pt,
                                                                1e-9))
        res = sim.get_eigenmode_coefficients(flux, [1])
        coeffs = res.alpha
        accumulated_flux_spectrum = mp.get_fluxes(flux)
        return coeffs, accumulated_flux_spectrum

    # Get incident flux coefficients
    if debug is True:
        source_mon_pt = mp.Vector3(x=source_x + 0.1)
        monsize = mp.Vector3(y=3*waveguide_width)
        source_fluxregion = mp.FluxRegion(center=source_mon_pt, size=monsize)
        src_coeffs = get_eigenmode_coeffs(sim, source_fluxregion, mon_pt)
        # ic(np.abs(src_coeffs[0, 0, 0])**2)

    # on no symmetry setting the np.abs(src_coeffs[0,0,0])**2 was previously
    # computed as
    abs_src_coeff = 57.97435797757672
    # but with symmetries=[mp.Mirror(direction=mp.Y)], we instead get
    # 31.632726229474404 ...

    # Get top output flux coefficients
    topmoncenter = mp.Vector3(size_x/2, arm_separation, 0)
    topfluxregion = mp.FluxRegion(topmoncenter, monsize)
    top_coeffs = get_eigenmode_coeffs(sim, topfluxregion, mon_pt)

    # fom1 = np.abs(top_coeffs[0, 0, 0])**2/np.abs(src_coeffs[0, 0, 0])**2
    fom1 = np.abs(top_coeffs[0, 0, 0])**2/abs_src_coeff
    if symmetry_enable:
        fom1 = fom1/2
    if debug is True:
        ic(fom1)
    return fom1


def meep_lumerical_comparison_experiment(num_samples=100):
    PATH = os.path.expanduser('~/scratch/nanophoto/lowfom/nodata/fields/')
    images = np.load(os.path.join(PATH, 'images.npy'),
                     mmap_mode='r')[:num_samples]
    foms = np.load(os.path.join(PATH, 'fom.npy'))[:num_samples]
    all_fom = []

    for image_idx in tqdm(range(num_samples)):
        # image_idx = 1
        image = images[image_idx]
        fom = compute_FOM(image)
        all_fom.append(fom)

    all_fom = np.stack(all_fom)
    np.save('meepfom.npy', all_fom)
    err = np.abs(foms - all_fom)
    _, axes = plt.subplots(1, 1)
    axes[0].hist(err, bins=10)
    plt.savefig(f'{image_idx}.png')


def _meep_get_fields(idx_map):
    assert idx_map.shape == (101, 91)
    sim, design_region, sim_args = get_sim()
    idx_map = double_with_mirror(idx_map)
    idx_map = normalise(idx_map)
    idx_map = mapping(idx_map, 0.5, 256, **sim_args)
    design_region.update_design_parameters(idx_map)

    fcen = sim_args['fcen']
    size = sim_args['design_region_size']
    # ic(size)
    # array champs de forme (190, 205)
    dx = 0.02
    sx = 189*dx,
    sy = 203*dx
    # ic(sx, sy)
    size = mp.Vector3(3.78, 4.07, 0)

    dft_monitor = sim.add_dft_fields(
        [mp.Ex, mp.Ey, mp.Ez],             # Components to monitor
        fcen, 0, 1,
        # frequency=fcen,                     # Operating frequency
        center=mp.Vector3(0, 0, 0),        # Center of the monitor region
        size=size          # Size of the monitor region
    )

    sim.run(until_after_sources=100)
    Ex = sim.get_dft_array(dft_monitor, mp.Ex, 0)
    Ey = sim.get_dft_array(dft_monitor, mp.Ey, 0)
    E = np.stack([Ex, Ey], axis=-1)
    # ic(Ex.shape, E.shape)

    return E


def meep_get_fields(images: np.array):
    """
    input:
        images: (B, 101, 92)
    output:
        fields: (B, 190, 206, 2) np.cpx
    """
    if images.ndim == 2:
        return _meep_get_fields(images)

    images = [images[i] for i in range(images.shape[0])]
    with multiprocessing.Pool() as pool:
        fields = pool.map(_meep_get_fields, images)
    fields = np.stack(fields)
    ic(fields.shape)
    # TODO: ajouter un catch de plantage de Meep
    return fields


if __name__ == '__main__':
    def test_meep_time(images):
        image = images[0]
        t0 = timeit.default_timer()
        fom = compute_FOM(image)
        t1 = timeit.default_timer()
        ic(fom)
        ic(t1-t0)

    def test_parallel_comp_scaling(images):
        # resultats sur le cluster mila avec 64 cpu
        # num_eval: temps
        # {2: 6.521025642752647,
        # 4: 6.3,
        # 8: 6.3,
        # 16: 6.3,
        # 32: 6.6,
        # 64: 8.2,
        # 128: 16.5}
        # ce qui donne environ 0.1 simulation/sec

        times = dict()
        for i in range(1, 8):
            num_eval = 2**i
            t0 = timeit.default_timer()
            foms = compute_FOM_parallele(images[:num_eval])
            t1 = timeit.default_timer()
            t = t1 - t0
            times[num_eval] = t
            ic(t)
            ic(foms)
        ic(times)

    def test_parallele_comp(images):
        fom = compute_FOM_parallele(images[:4])
        ic(fom)

    def test_array_comp(images):
        foms = compute_FOM_array(images[:4])
        ic(foms)

    def test_symmetrie(images):
        for i in range(3):
            print('sym true')
            t0 = timeit.default_timer()
            fom0 = compute_FOM(images[i], symmetry_enable=True)
            t1 = timeit.default_timer()
            ic(t1-t0)
            print('sym false')
            t0 = timeit.default_timer()
            fom1 = compute_FOM(images[i], symmetry_enable=False)
            t1 = timeit.default_timer()
            ic(t1-t0)
        assert fom0 == fom1

    def test_get_fields(image):
        fields = meep_get_fields(image)
        fxr = np.real(fields[..., 0])
        fxi = np.imag(fields[..., 0])
        fyr = np.real(fields[..., 1])
        fyi = np.imag(fields[..., 1])
        fields_list = [fxr, fxi, fyr, fyi]
        _, axes = plt.subplots(2, 2)
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(fields_list[i])
            ax.axis('off')
        plt.savefig('fields_test.png')
        ic('ok')

    def test_get_fields_parallel(images):
        E = meep_get_fields(images)
        _, axes = plt.subplots(2, 2)
        E = np.real(E[..., 0])
        axes = axes.flatten()
        for i in range(4):
            axes[i].imshow(E[i])
            axes[i].axis('off')
        plt.savefig('test_fields_parallel.png')

    def test_compute_fom(image):
        fom = compute_FOM(image)
        ic(fom)

    def test_get_FOM_and_gradient(image):
        fom, grad = compute_FOM_and_gradient(image)
        ic(fom)
        ic(grad.shape, grad.min(), grad.max(), grad.mean())

    PATH = os.path.expanduser('~/scratch/nanophoto/lowfom/nodata/fields/')
    images = np.load(os.path.join(PATH, 'images.npy'), mmap_mode='r')

    # test_get_fields_parallel(images[:4])
    # test_get_fields(images[0])
    # test_symmetrie(images)
    # test_array_comp(images)
    test_parallele_comp(images)
    # test_parallel_comp_scaling(images)
    # test_meep_time()
    # test_compute_fom(images[0])
    # test_get_FOM_and_gradient(images[0])
