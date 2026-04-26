import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from pymatgen.io.vasp import Chgcar
from scipy.interpolate import RegularGridInterpolator
import os
import math


def process_single_structure(contcar_path, elfcar_path, output_npz_path, search_radius=3.0, plot_images=False,
                             output_dir="elf_plots"):
    """
    遍历整个超胞，提取所有化学键截面（包含周期性扩充与多角度旋转），并保存为机器学习专用的二进制特征库
    """
    if plot_images and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"正在读取结构文件: {contcar_path}")
    structure = Structure.from_file(contcar_path)
    total_atoms = len(structure)
    print(f"识别到超胞中包含 {total_atoms} 个原子。")

    elf_data = Chgcar.from_file(elfcar_path)
    lattice = structure.lattice.matrix

    # ==========================================
    # 核心修改 1：Padding 与坐标轴同步扩充
    # ==========================================
    grid = elf_data.data['total']
    nx, ny, nz = grid.shape
    # 动态将扩充层数设定为该维度的网格总数（等于向外各扩展一个完整原胞）
    pad_x, pad_y, pad_z = nx, ny, nz

    # 分别在 X, Y, Z 三个方向进行周期性扩充
    padded_grid = np.pad(grid, pad_width=((pad_x, pad_x), (pad_y, pad_y), (pad_z, pad_z)), mode='wrap')

    # 计算原网格的真实步长
    dx, dy, dz = 1.0 / nx, 1.0 / ny, 1.0 / nz

    # 构建包含边界外区域的全新坐标轴
    x = np.arange(-pad_x, nx + pad_x) * dx
    y = np.arange(-pad_y, ny + pad_y) * dy
    z = np.arange(-pad_z, nz + pad_z) * dz

    # 使用同步扩充后的坐标和数据建立插值引擎
    interpolator = RegularGridInterpolator((x, y, z), padded_grid, bounds_error=False, fill_value=0.0, method='cubic')

    # 用于存储机器学习特征的数据池
    all_elf_matrices = []
    all_metadata = []
    processed_bonds = set()

    print(f"开始遍历提取化学键 (搜索半径: {search_radius} Å)...")

    for atom_idx in range(total_atoms):
        neighbors = structure.get_neighbors(structure[atom_idx], r=search_radius)

        for nn in neighbors:
            nn_idx = nn.index

            if atom_idx < nn_idx:
                bond_pair = (atom_idx, nn_idx)
                if bond_pair not in processed_bonds:
                    processed_bonds.add(bond_pair)

                    # 获取坐标和键长信息
                    site1, site2 = structure[atom_idx], structure[nn_idx]
                    p1, p2 = site1.coords, site2.coords
                    dist, image = site1.distance_and_image(site2)
                    p2 = p2 + np.dot(image, lattice)

                    bond_vector = p2 - p1
                    bond_length = np.linalg.norm(bond_vector)
                    bond_center = (p1 + p2) / 2.0

                    u = bond_vector / bond_length

                    # 生成初始的法向量 v0
                    tmp_vec = np.array([1, 0, 0]) if np.abs(u[0]) < 0.9 else np.array([0, 1, 0])
                    v0 = np.cross(u, tmp_vec)
                    v0 = v0 / np.linalg.norm(v0)

                    margin = 0.0
                    len_u = bond_length + 2 * margin
                    len_v = bond_length / 2
                    resolution = 200

                    grid_u = np.linspace(-len_u / 2, len_u / 2, resolution)
                    grid_v = np.linspace(-len_v / 2, len_v / 2, int(resolution * (len_v / len_u)))
                    U, V = np.meshgrid(grid_u, grid_v)
                    inv_lattice = np.linalg.inv(lattice)

                    # ==========================================
                    # 核心修改 2：加入旋转角度遍历 (0, 60, 120 度)
                    # ==========================================
                    angles_deg = [0, 60, 120]

                    for angle in angles_deg:
                        theta = math.radians(angle)
                        # 利用罗德里格旋转公式，让 v0 绕着 u 轴旋转 theta 角度
                        v_rot = v0 * math.cos(theta) + np.cross(u, v0) * math.sin(theta)

                        points_cartesian = bond_center + U[..., np.newaxis] * u + V[..., np.newaxis] * v_rot

                        # ==========================================
                        # 核心修改 3：去掉 % 1.0 (让超出边界的坐标直接读取 Padding 区)
                        # ==========================================
                        points_fractional = np.dot(points_cartesian, inv_lattice)

                        # 提取二维 ELF 数据矩阵
                        elf_2d = interpolator(points_fractional)

                        meta = {
                            "atom_idx_1": atom_idx,
                            "atom_idx_2": nn_idx,
                            "element_1": site1.species_string,
                            "element_2": site2.species_string,
                            "bond_length": bond_length,
                            "rotation_angle": angle  # 记录旋转角度
                        }
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

                            # 图片名加上角度标识
                            filename = f"{output_dir}/{site1.species_string}{atom_idx}_{site2.species_string}{nn_idx}_{angle}deg_ELF.png"
                            plt.tight_layout()
                            plt.savefig(filename, dpi=150, bbox_inches='tight')
                            plt.close()

    print(f"总共提取了 {len(all_elf_matrices)} 个切片特征。")
    print(f"正在写入二进制文件: {output_npz_path} ...")

    np.savez_compressed(output_npz_path,
                        features=np.array(all_elf_matrices, dtype=object),
                        metadata=np.array(all_metadata, dtype=object))
    print("任务结束！")


if __name__ == "__main__":
    contcar_file = "1CoCrFeNi/00_SQS_Base/CONTCAR"
    elfcar_file = "1CoCrFeNi/00_SQS_Base/ELFCAR"
    output_binary_name = "1CoCrFeNi/00_SQS_Base/HEA_single_config_features.npz"

    process_single_structure(contcar_file, elfcar_file, output_binary_name,
                             search_radius=3.0, plot_images=True, output_dir="1CoCrFeNi/00_SQS_Base/test_images_001")