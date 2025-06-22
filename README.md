# 🧠 3D Content Creation Optimization Using Subject-Driven Textual Prompts

This project explores the use of **subject-driven textual guidance** to optimize 3D content generation using the [Make-It-3D](https://github.com/junshutang/Make-It-3D) model. It investigates how the richness, perspective, and specificity of natural language prompts influence the visual fidelity and semantic alignment of generated 3D meshes.

---

## 📌 Project Overview

- **Goal**: Improve 3D mesh generation quality by customizing the input prompts according to subject-specific perspectives.
- **Model Used**: [Make-It-3D](https://github.com/junshutang/Make-It-3D)
- **Hardware**: NVIDIA A100 GPU
- **Main Contribution**: Introduced five structured subject-driven prompt types and analyzed their impact using CLIP-based alignment scoring.

---

## 🛠️ Work Done

- **Environment Setup**: Installed and configured the Make-It-3D model.
- **Prompt Categories**:
  - 🧑‍🎨 Artist
  - 🛠️ Craftsperson
  - 👁️ Normal Observer
  - 📐 Expert Descriptor
  - 🧑‍💻 Designer
- **Objects Generated**: Teddy Bear, Minion, Sword, Water Splash, Bonsai Tree, Wine Bottle, Gold Vase
- **Mesh Generation**: 200 Teddy bears + 150 other objects across various prompt types and complexity levels.

---

## 🧪 Methodology

**Example Teddy Bear Mesh Generation**:
```bash
python main_5.py --workspace teddy_0005 \
--ref_path "demo/teddy.png" \
--phi_range 135 225 \
--iters 2000 \
--fov 60 \
--fovy_range 50 70 \
--blob_radius 0.2 \
--text "A teddy bear with a round face" \
--refine_iters 300 \
--refine \
--save_mesh
