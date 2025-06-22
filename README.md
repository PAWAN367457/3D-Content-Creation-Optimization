
# 3D Content Creation Optimization

This project explores the optimization of 3D mesh generation through structured, subject-driven textual prompts. Built on top of the [Make-It-3D](https://github.com/junshutang/Make-It-3D) model, the work investigates how prompt specificity, perspective, and language richness affect the semantic alignment and visual quality of generated 3D assets.

## Overview

- **Objective**: Analyze the role of prompt design in improving 3D generation quality.
- **Model Used**: Make-It-3D (pretrained model using neural implicit surface representations).
- **Focus**: Evaluate mesh-text alignment using CLIP-based similarity scoring.
- **Contributions**:
  - Introduced five structured prompt perspectives (Artist, Craftsperson, Observer, Expert, Designer).
  - Generated 350+ 3D meshes with varied text inputs.
  - Conducted quantitative and qualitative evaluation.

## Methodology

Each object was generated using custom text descriptions across five perspectives. Prompt categories were:

- Artist: Focused on visual and emotional elements.
- Craftsperson: Focused on material quality and technical details.
- Normal Observer: General impression and simple description.
- Expert Descriptor: Detailed multi-dimensional visual analysis.
- Designer: Combined form, function, and usability.

### Sample Generation Command

Teddy bear (2000 iterations):
```bash
python main_5.py --workspace teddy_0005 \
--ref_path "demo/teddy.png" \
--phi_range 135 225 \
--iters 2000 \
--fov 60 \
--fovy_range 50 70 \
--blob_radius 0.2 \
--text "A teddy bear with a round face" \
--refine_iters 300 --refine --save_mesh
```

Other objects (1000 iterations):
```bash
python main_1.py --workspace sword_E_3 \
--ref_path "demo/sword.png" \
--phi_range 135 225 \
--iters 1000 \
--fov 90 \
--fovy_range 45 110 \
--blob_radius 0.2 \
--text "Blade shows forge marks beneath polished surface finish" \
--refine_iters 200 --refine --save_mesh
```

## Results Summary

Prompt categories that combined structure and clarity performed best. Designer and Expert prompts yielded the most coherent meshes. Observations:

- Prompts with higher detail and structure scored higher on CLIP alignment.
- Object-specific trends were identified (e.g., simple prompts worked better for organic shapes like bonsai trees).
- Designer prompts had the highest average alignment scores across all tested objects.

## Limitations

- Mesh generation was computationally expensive (20–50 minutes per mesh).
- Prompt quality heavily influenced results; overly abstract or vague prompts underperformed.
- CLIP-based scoring may not fully reflect human perception.

## Authors

- Nitin Kumar Das (2024csm1014)
- Pawan Kumar (2024aim1007)

## References

- Make-It-3D: https://github.com/junshutang/Make-It-3D
- CLIP: Contrastive Language–Image Pre-training
