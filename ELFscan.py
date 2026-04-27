import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from pymatgen.io.vasp import Chgcar  # ELFCAR 格式与 CHGCAR 完全一致
from scipy.interpolate import RegularGridInterpolator
import os

def extract_bond_elf(contcar_path, elfcar_path, atom_idx_1, atom_idx_2, output_dir="elf_plots"):
    """
    提取任意两个原子之间的二维 ELF 截面并绘图
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 读取结构和 3D 体数据
    structure = Structure.from_file(contcar_path)
    # pymatgen 用 Chgcar 类来读取 ELFCAR，因为它们的数据格式是相同的
    elf_data = Chgcar.from_file(elfcar_path)

    # 提取晶格矩阵和 3D 网格数据
    lattice = structure.lattice.matrix
    # 注意：ELFCAR 的数据可能需要转置以匹配 (x, y, z) 坐标轴
    grid = elf_data.data['total']

    nx, ny, nz = grid.shape
    # 构建 3D 网格的坐标系（分数坐标 0 到 1）
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)

    # 构建插值器 (处理周期性边界条件需要特别注意，这里为了演示使用简单插值)
    # 对于高熵合金大超胞，通常键在内部，如果跨越边界需要做平移处理
    interpolator = RegularGridInterpolator((x, y, z), grid, bounds_error=False, fill_value= 0.0,method = 'cubic')

    # 2. 获取两个原子的笛卡尔坐标
    site1 = structure[atom_idx_1]
    site2 = structure[atom_idx_2]
    p1 = site1.coords
    p2 = site2.coords

    # 处理周期性边界条件（如果两个原子跨越了原胞边界，获取它们最短距离的映射）
    dist, image = site1.distance_and_image(site2)
    p2 = p2 + np.dot(image, lattice)  # 修正跨边界的 p2 坐标

    # 3. 建立二维截面 (以键长为中心，向两侧扩展)
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

    # ================= 设定图像的物理尺寸 自适应截面尺寸与边缘留白 =================
    # 定义“留白距离 (margin)”：也就是在原子外侧再额外扫描多少距离（埃）
    # 设置为 1.5，意味着左边原子的左侧（负坐标）和右边原子的右侧，都会额外多出 1.5 埃的图像
    margin = 0.0

    # 沿着键的方向 (u轴) 的总长度 = 真实的键长 + 两端的留白距离
    len_u = bond_length + 2 * margin

    # 垂直于键的方向 (v轴) 的总宽度。考虑到高熵合金原子的电子云范围，5.0 埃是一个比较舒适的宽度
    #len_v = 5.0
    len_v = bond_length/2
    # ========================================================================


    resolution = 200  # 100x100 的像素网格 256

    grid_u = np.linspace(-len_u / 2, len_u / 2, resolution)
    grid_v = np.linspace(-len_v / 2, len_v / 2, resolution)
    U, V = np.meshgrid(grid_u, grid_v)

    # 4. 将二维网格映射回三维笛卡尔坐标，再转为分数坐标进行插值
    # shape: (resolution, resolution, 3)
    points_cartesian = bond_center + U[..., np.newaxis] * u + V[..., np.newaxis] * v

    # 将笛卡尔坐标转为分数坐标以用于 interpolator
    inv_lattice = np.linalg.inv(lattice)
    points_fractional = np.dot(points_cartesian, inv_lattice)

    # 将分数坐标限制在 [0, 1) 区间内，处理周期性
    points_fractional = points_fractional % 1.0

    # 提取二维 ELF 数据
    elf_2d = interpolator(points_fractional)

    # 5. 绘制二维 Contour 图
    # 调整画布比例，留出放 Colorbar 的空间
    fig, ax = plt.subplots(figsize=(6, 5))

    # ================= 核心修改区：人工缩小标尺 =================
    # 强制生成 0.0 到 0.3 的 100 个等高线层级 (如果你觉得图不够红，可以把0.3改成0.25或0.2)
    custom_levels = np.linspace(0.0, 0.3, 101)

    # 传入 custom_levels，并设置 extend='max'
    # 注意：这里去掉了原来的 vmin 和 vmax 参数
    contour = ax.contourf(U, V, elf_2d,
                          levels=custom_levels,
                          cmap='jet',
                          extend='max')

    # 画出两个原子的位置
    ax.plot([-bond_length / 2, bond_length / 2], [0, 0], 'ko', markersize=8)

    ax.axis('off')  # 隐藏坐标轴边框和刻度

    # 添加侧边 ELF 数值图例 (Colorbar)
    # ticks 参数强制规定了图例旁边的数字显示 0.0, 0.06, 0.12, 0.18, 0.24, 0.30
    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04, ticks=np.linspace(0.0, 0.3, 6))
    cbar.set_label('ELF value', rotation=270, labelpad=15, fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    # ============================================================
    ele1 = site1.species_string
    ele2 = site2.species_string
    filename = f"{output_dir}/{ele1}{atom_idx_1}_{ele2}{atom_idx_2}_ELF.png"

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


# ================= 批处理逻辑 =================
''''# 在你的实际任务中，你需要用 pymatgen 的 CrystalNN 来找到你要的键
def batch_process_all_bonds(contcar_path, elfcar_path):
    structure = Structure.from_file(contcar_path)
    # 这里为了演示，假设我们只提取第 0 号原子和它最近邻的第 1 号原子的键
    # 实际应用中，你应该遍历 structure 的近邻列表，提取你需要的特定元素对 (如 Co-Ni)

    # 示例调用：
    extract_bond_elf(contcar_path, elfcar_path, atom_idx_1=0, atom_idx_2=1) '''


def batch_process_all_bonds(contcar_path, elfcar_path):
    structure = Structure.from_file(contcar_path)

    # 🌟 修改这里：输入你想测试的原子索引 (范围 0 到 99)
    target_atom = 25

    print(f"正在分析第 {target_atom} 号 ({structure[target_atom].species_string}) 及其近邻...")

    # 获取该原子周围半径 3.0 埃以内的所有原子 (对于高熵合金，第一配位层通常在 2.5 埃左右)
    # r=3.0 是一个安全的截断半径，能保证抓到近邻原子
    neighbors = structure.get_neighbors(structure[target_atom], r=6.0)

    if len(neighbors) == 0:
        print("在这个半径内没有找到近邻原子，请检查坐标或调大半径。")
        return

    # 获取它的第一个邻居（距离最近的原子之一）
    neighbor_site = neighbors[0]
    neighbor_idx = neighbor_site.index  # 获取邻居在 structure 中的真实索引

    print(
        f"找到最近邻: 第 {neighbor_idx} 号原子 ({neighbor_site.species_string})，距离: {neighbor_site.nn_distance:.3f} Å")

    # 提取并绘制这对原子的 ELF 图
    extract_bond_elf(contcar_path, elfcar_path, atom_idx_1=target_atom, atom_idx_2=neighbor_idx)

# 运行示例
batch_process_all_bonds("CONTCAR", "ELFCAR")
