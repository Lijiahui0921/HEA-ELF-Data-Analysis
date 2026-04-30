import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from pymatgen.io.vasp import Chgcar
from scipy.interpolate import RegularGridInterpolator
import os
import math


# =====================================================================
# 核心扫描函数
# =====================================================================
def process_single_structure(contcar_path, elfcar_path, output_npz_path, search_radius=3.0, plot_images=False,
                             image_output_dir="elf_plots"):
    """遍历整个超胞，提取所有化学键截面，并保存为机器学习专用的二进制特征库。包含：大尺寸扫描视野、对数空间映射消除负数。"""
    if plot_images:
        os.makedirs(image_output_dir, exist_ok=True)

    structure = Structure.from_file(contcar_path)
    total_atoms = len(structure)

    elf_data = Chgcar.from_file(elfcar_path)
    lattice = structure.lattice.matrix

    # 数学处理：无穷空间映射
    grid = elf_data.data['total']

    # 1. 截断保护：防止出现 0 或 1 导致除以零或对数域无意义
    epsilon = 1e-12
    grid_clipped = np.clip(grid, epsilon, 1.0 - epsilon)

    # 2. 空间映射：将 (0,1) 的 ELF 映射到 (-无穷, +无穷) 的 y 空间
    # 公式：y = ln(1/ELF - 1)
    grid_y = np.log(1.0 / grid_clipped - 1.0)

    # Padding 与插值 (在 y 空间进行)
    nx, ny, nz = grid_y.shape
    pad_x, pad_y, pad_z = nx, ny, nz

    padded_grid_y = np.pad(grid_y, pad_width=((pad_x, pad_x), (pad_y, pad_y), (pad_z, pad_z)), mode='wrap')

    dx, dy, dz = 1.0 / nx, 1.0 / ny, 1.0 / nz

    x = np.arange(-pad_x, nx + pad_x) * dx
    y = np.arange(-pad_y, ny + pad_y) * dy
    z = np.arange(-pad_z, nz + pad_z) * dz

    #此时插值引擎插值的是无边界的 y 值
    # 设定 fill_value=np.log(1.0/epsilon - 1.0) 意味着超出边界的地方默认 ELF=0
    interpolator = RegularGridInterpolator((x, y, z), padded_grid_y, bounds_error=False,
                                           fill_value=np.log(1.0 / epsilon - 1.0), method='cubic')

    all_elf_matrices = []
    all_metadata = []
    processed_bonds = set()

    for atom_idx in range(total_atoms):
        neighbors = structure.get_neighbors(structure[atom_idx], r=search_radius)

        for nn in neighbors:
            nn_idx = nn.index

            if atom_idx < nn_idx:
                bond_pair = (atom_idx, nn_idx)
                if bond_pair not in processed_bonds:
                    processed_bonds.add(bond_pair)

                    site1, site2 = structure[atom_idx], structure[nn_idx]
                    p1, p2 = site1.coords, site2.coords
                    dist, image = site1.distance_and_image(site2)
                    p2 = p2 + np.dot(image, lattice)

                    bond_vector = p2 - p1
                    bond_length = np.linalg.norm(bond_vector)
                    bond_center = (p1 + p2) / 2.0

                    u = bond_vector / bond_length

                    tmp_vec = np.array([1, 0, 0]) if np.abs(u[0]) < 0.9 else np.array([0, 1, 0])
                    v0 = np.cross(u, tmp_vec)
                    v0 = v0 / np.linalg.norm(v0)

                    # ---------------------------------------------------------
                    # 【尺寸修改：扩大扫描面，使其覆盖更广的环境】
                    # ---------------------------------------------------------
                    margin = 0.0  # 增加向外的延伸余量 (Å)
                    len_u = bond_length + 2 * margin
                    len_v = len_u  # 让长宽一致，获取正方形的宽广视野截面
                    resolution = 200

                    grid_u = np.linspace(-len_u / 2, len_u / 2, resolution)
                    grid_v = np.linspace(-len_v / 2, len_v / 2, int(resolution * (len_v / len_u)))
                    U, V = np.meshgrid(grid_u, grid_v)
                    inv_lattice = np.linalg.inv(lattice)

                    angles_deg = [0, 60, 120]

                    for angle in angles_deg:
                        theta = math.radians(angle)
                        v_rot = v0 * math.cos(theta) + np.cross(u, v0) * math.sin(theta)

                        points_cartesian = bond_center + U[..., np.newaxis] * u + V[..., np.newaxis] * v_rot
                        points_fractional = np.dot(points_cartesian, inv_lattice)

                        # 1. 提取无限空间的三次插值结果 y_2d
                        y_2d = interpolator(points_fractional)

                        # 2. 【核心逆变换：将 y_2d 反推回 ELF 物理域 (0,1)】
                        # 使用 np.float64 保证极高精度，避免浮点溢出警告
                        elf_2d = 1.0 / (1.0 + np.exp(np.float64(y_2d)))

                        meta = {
                            "atom_idx_1": atom_idx,
                            "atom_idx_2": nn_idx,
                            "element_1": site1.species_string,
                            "element_2": site2.species_string,
                            "bond_length": bond_length,
                            "rotation_angle": angle
                        }

                        # 数据池中记录的是完美且符合物理意义的反推后数据
                        all_elf_matrices.append(elf_2d)
                        all_metadata.append(meta)

                        if plot_images:
                            fig, ax = plt.subplots(figsize=(6, 5))
                            v_min, v_max = np.min(elf_2d), np.max(elf_2d)
                            custom_levels = np.linspace(v_min, v_max, 100)

                            contour = ax.contourf(U, V, elf_2d, levels=custom_levels, cmap='jet')
                            ax.axis('off')
                            cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
                            cbar.set_label('ELF value', rotation=270, labelpad=15, fontsize=12)
                            cbar.ax.tick_params(labelsize=10)

                            filename = f"{image_output_dir}/{site1.species_string}{atom_idx}_{site2.species_string}{nn_idx}_{angle}deg_ELF.png"
                            plt.tight_layout()
                            plt.savefig(filename, dpi=150, bbox_inches='tight')
                            plt.close()

    # 确保存储 NPZ 的外层文件夹存在
    os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)
    np.savez_compressed(output_npz_path,
                        features=np.array(all_elf_matrices, dtype=object),
                        metadata=np.array(all_metadata, dtype=object))


# =====================================================================
# 批处理逻辑模块
# =====================================================================
def run_batch_processing(base_dir, enable_plotting=False, target_heas=None, max_swaps=None):
    """
    增加了测试控制参数：
    - target_heas: list，例如 ["1CoCrFeNi"]。如果提供，只处理列表内的体系。
    - max_swaps: int，例如 2。如果提供，则限制每个体系下处理的 Random_Swap 文件夹数量。
    """
    FEATURE_SUBFOLDER_NAME = "Features"

    print(f"正在扫描主目录: {base_dir}")

    hea_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

    for hea_name in hea_folders:
        # 【新增逻辑】：如果指定了测试的体系且当前体系不在列表中，则直接跳过
        if target_heas is not None and hea_name not in target_heas:
            continue

        hea_path = os.path.join(base_dir, hea_name)

        sqs_base_dir = os.path.join(hea_path, "00_SQS_Base")
        swaps_dir = os.path.join(hea_path, "01_Random_Swaps")

        hea_feature_output_dir = os.path.join(hea_path, FEATURE_SUBFOLDER_NAME)

        if not os.path.exists(sqs_base_dir) and not os.path.exists(swaps_dir):
            continue

        print(f"\n========== 体系: {hea_name} ==========")

        os.makedirs(hea_feature_output_dir, exist_ok=True)

        if os.path.exists(sqs_base_dir):
            _check_and_process_folder(sqs_base_dir, hea_name, "00_SQS_Base",
                                      hea_feature_output_dir, enable_plotting)

        if os.path.exists(swaps_dir):
            swap_subfolders = sorted([f for f in os.listdir(swaps_dir) if os.path.isdir(os.path.join(swaps_dir, f))])

            # 【新增逻辑】：如果指定了 max_swaps，对列表进行切片截取
            if max_swaps is not None:
                swap_subfolders = swap_subfolders[:max_swaps]
                print(f"[*] 测试模式：仅处理前 {max_swaps} 个 Random_Swap 文件夹。")

            for sub_folder in swap_subfolders:
                target_dir = os.path.join(swaps_dir, sub_folder)
                _check_and_process_folder(target_dir, hea_name, f"Random_Swap_{sub_folder}",
                                          hea_feature_output_dir, enable_plotting)


def _check_and_process_folder(target_dir, hea_name, config_name, output_npz_dir, enable_plotting):
    contcar_path = os.path.join(target_dir, "CONTCAR")
    elfcar_path = os.path.join(target_dir, "ELFCAR")

    output_npz_path = os.path.join(output_npz_dir, f"{hea_name}_{config_name}_features.npz")
    image_output_dir = os.path.join(target_dir, "elf_plots")

    if os.path.exists(output_npz_path):
        print(f"[*] 跳过: {config_name} (已存在)")
        return

    if os.path.exists(contcar_path) and os.path.exists(elfcar_path):
        print(f"[->] 正在处理: {hea_name} / {config_name} ...")
        try:
            process_single_structure(contcar_path, elfcar_path, output_npz_path,
                                     search_radius=3.0, plot_images=enable_plotting,
                                     image_output_dir=image_output_dir)
            print(f"    [√] 数据已存至 Features 文件夹")
            if enable_plotting:
                print(f"    [√] 图像已就地存至 {image_output_dir}")
        except Exception as e:
            print(f"    [!] 错误: {e}")
    else:
        print(f"[-] 缺失数据: {config_name}")


# =====================================================================
# 运行入口
# =====================================================================
if __name__ == "__main__":
    MAIN_DATA_DIR = r"E:\HEAs_Data"

    print("启动测试模式...")
    # [测试示例]
    # 只跑 1CoCrFeNi 这一个体系，并且只跑前 2 个 Random_Swap 文件夹
    run_batch_processing(
        base_dir=MAIN_DATA_DIR,
        enable_plotting=True,  # 测试时建议开启画图来检查结果
        target_heas=["1CoCrFeNi"],  # 只处理这个文件夹
        max_swaps=0  # 限制 swap 数量为 2 (设为 0 则只跑 SQS_Base)
    )

    print("\n任务执行完毕！")