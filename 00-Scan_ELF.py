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
    遍历整个超胞，提取所有化学键截面，并保存为机器学习专用的二进制特征库
    """
    if plot_images and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"正在读取结构文件: {contcar_path}")
    structure = Structure.from_file(contcar_path)
    total_atoms = len(structure)
    print(f"识别到超胞中包含 {total_atoms} 个原子。")

    #print(f"正在将 ELFCAR 加载到内存并构建 3D 插值引擎 (这可能需要几十秒)...")
    elf_data = Chgcar.from_file(elfcar_path)
    lattice = structure.lattice.matrix
    grid = elf_data.data['total']

    nx, ny, nz = grid.shape
    x, y, z = np.linspace(0, 1, nx), np.linspace(0, 1, ny), np.linspace(0, 1, nz)
    interpolator = RegularGridInterpolator((x, y, z), grid, bounds_error=False, fill_value=0.0, method='cubic')
    #print(f"✅ 3D 插值引擎构建完成！")

    # 用于存储机器学习特征的数据池
    all_elf_matrices = []
    all_metadata = []
    processed_bonds = set()  # 排重集合

    print(f"开始遍历提取化学键 (搜索半径: {search_radius} Å)...")

    # 遍历每一个原子
    for atom_idx in range(total_atoms):
        # 寻找指定半径内的所有邻居
        neighbors = structure.get_neighbors(structure[atom_idx], r=search_radius)

        for nn in neighbors:
            nn_idx = nn.index

            # 【核心排重逻辑】：无向图遍历，确保 A-B 键和 B-A 键只提取一次
            if atom_idx < nn_idx:
                bond_pair = (atom_idx, nn_idx)
                if bond_pair not in processed_bonds:
                    processed_bonds.add(bond_pair)

                    # --- 开始几何切片核心逻辑 ---
                    # 获取两个原子的笛卡尔坐标
                    site1, site2 = structure[atom_idx], structure[nn_idx]
                    p1, p2 = site1.coords, site2.coords
                    dist, image = site1.distance_and_image(site2)
                    p2 = p2 + np.dot(image, lattice)
                    # 建立二维截面 (以键长为中心，向两侧扩展)
                    bond_vector = p2 - p1
                    bond_length = np.linalg.norm(bond_vector)
                    bond_center = (p1 + p2) / 2.0

                    # 定义截面的方向：
                    # u 向量：沿着键的方向
                    u = bond_vector / bond_length
                    # 寻找一个与 u 垂直的 v 向量 (为了确定二维平面的朝向)
                    # 随机找一个不与 u 共线的向量进行叉乘
                    tmp_vec = np.array([1, 0, 0]) if np.abs(u[0]) < 0.9 else np.array([0, 1, 0])
                    v = np.cross(u, tmp_vec)
                    v = v / np.linalg.norm(v)

                    # 设置图像的物理尺寸参数
                    margin = 0.0
                    len_u = bond_length + 2 * margin
                    len_v = bond_length / 2
                    resolution = 200  #像素网格200×200

                    grid_u = np.linspace(-len_u / 2, len_u / 2, resolution)
                    grid_v = np.linspace(-len_v / 2, len_v / 2, int(resolution * (len_v / len_u)))  # 保持像素正方形
                    U, V = np.meshgrid(grid_u, grid_v)

                    points_cartesian = bond_center + U[..., np.newaxis] * u + V[..., np.newaxis] * v
                    # 将笛卡尔坐标转为分数坐标以用于 interpolator
                    inv_lattice = np.linalg.inv(lattice)
                    points_fractional = np.dot(points_cartesian, inv_lattice) % 1.0     # 将分数坐标限制在 [0, 1) 区间内，处理周期性

                    # 提取二维 ELF 数据矩阵
                    elf_2d = interpolator(points_fractional)

                    # --- 记录数据字典 ---
                    meta = {
                        "atom_idx_1": atom_idx,
                        "atom_idx_2": nn_idx,
                        "element_1": site1.species_string,
                        "element_2": site2.species_string,
                        "bond_length": bond_length
                    }
                    all_elf_matrices.append(elf_2d)
                    all_metadata.append(meta)

                    # --- 可选：画图 ---
                    if plot_images:
                        fig, ax = plt.subplots(figsize=(6, 5))

                        # 动态获取当前截面的最大最小值，稍微留出余量
                        v_min = np.min(elf_2d)
                        v_max = np.max(elf_2d)

                        # 自动生成 100 个等高线层级，适应当前数据，避免白色空洞
                        custom_levels = np.linspace(v_min, v_max, 100)

                        # 去掉 vmin 和 vmax 限制，让 colormap 自动拉伸
                        contour = ax.contourf(U, V, elf_2d, levels=custom_levels, cmap='jet')

                        ax.axis('off')

                        # 自动设置 colorbar 的 ticks，使其整洁
                        cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
                        cbar.set_label('ELF value', rotation=270, labelpad=15, fontsize=12)
                        cbar.ax.tick_params(labelsize=10)

                        filename = f"{output_dir}/{site1.species_string}{atom_idx}_{site2.species_string}{nn_idx}_ELF.png"
                        plt.tight_layout()
                        plt.savefig(filename, dpi=150, bbox_inches='tight')
                        plt.close()

    # --- 将所有数据打包保存为二进制 .npz 文件 ---
    print(f"总共提取了 {len(all_elf_matrices)} 条独立化学键的数据。")
    print(f"正在将特征矩阵写入二进制文件: {output_npz_path} ...")

    # 将 list 转换为统一的 numpy 多维数组。注意：由于键长不同，矩阵尺寸可能不同，这里使用 object 数组保存，或在外部 padding。
    # 为了保证 numpy 存储的兼容性，直接存 list 字典
    np.savez_compressed(output_npz_path,
                        features=np.array(all_elf_matrices, dtype=object),
                        metadata=np.array(all_metadata, dtype=object))
    print("任务结束！")


# ========================================================
# 运行示例
# ========================================================
if __name__ == "__main__":
    # 你的文件路径
    contcar_file = "1CoCrFeNi/00_SQS_Base/CONTCAR"
    elfcar_file = "1CoCrFeNi/00_SQS_Base/ELFCAR"

    # 指定输出的二进制文件名称
    output_binary_name = "1CoCrFeNi/00_SQS_Base/HEA_single_config_features.npz"

    # 执行处理
    # plot_images=False 表示纯粹提取数据，不画图，速度最快
    # search_radius=3.0 是典型的第一配位层截止半径，保证每个原子能抓到 8~12 条键
    process_single_structure(contcar_file, elfcar_file, output_binary_name,
                             search_radius=3.0, plot_images=True, output_dir="1CoCrFeNi/00_SQS_Base/test_images_001")