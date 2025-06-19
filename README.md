# Response to Reviewer ERPT
Thank you very much for your valuable comments. We clarify as follows:

**1. Analysis of α**

We set $\alpha=1$ to balance losses between RGB and IR modalities, ensuring joint gradient-driven optimization, rather than dominated by one. A supplementary experiment on the "apples" scene (Table below) confirms that α=1 outperforms one-modality-dominant alternatives (e.g., 0.1 or 10). In the revised manuscript, we will add a new subsection “5.6 Parameter Sensitivity Analysis” in the “5 Experiments” section to further add sensitivity analysis on $\alpha$, showing the performance changes of the model under different $\alpha$.
|Modality|Metric||$\alpha$||
|-|-|-|-|-|
|||0.1|**1**|10|
|IR|PSNR↑|43.49|**45.13**|43.58|
||SSIM↑|0.992|**0.994**|0.992|
||LPIPS↓|0.097|**0.090**|0.091|
|RGB|PSNR↑|39.51|**40.40**|39.95|
||SSIM↑|0.985|**0.988**|0.987|
||LPIPS↓|0.086|**0.065**|0.069|

**2. Improvement for RGB Modality**

The baseline method for the RGB modality—Mip-Splatting already achieves high rendering quality (both PSNR and SSIM are strong). Therefore, the room for improvement is relatively limited. Nevertheless, BMGS is still able to suppress "floating" or unstable Gaussians by leveraging cross-modal geometric constraints. As shown in Fig.6 of our paper, under the same viewpoint, Mip-Splatting exhibits slight floating artifacts, whereas BMGS suppresses these using geometry shared across modalities, resulting in more detailed rendering.

**3. Ablation Analysis of Calibration Module**

To confirm the calibration module's critical role, we performed ablation studies assessing reprojection error and reconstruction quality:

**(1) Reprojection Error**: Removing the calibration module drastically increases reprojection error to **~12 px**, indicating severe misalignment. Using BMGS calibration achieves a sub-pixel accuracy of **~0.21 px**.
|Method|||REproj|||
|-|-|-|-|-|-|
||pair1|pair2|pair3|pair4|Avg|
|w/o Calib|12.32|12.29|12.05|11.97|12.16|
|**BMGS**|**0.25**|**0.25**|**0.14**|**0.21**|**0.21**|


**(2) Reconstruction Performance**: Without calibration, the "apples" scene experiences significant performance degradation (IR: -1.64 dB, RGB: -0.76 dB PSNR), highlighting calibration’s necessity.
|Modality|Method|PSNR↑|SSIM↑|LPIPS↓|
|-|-|-|-|-|
|IR|w/o Calib|43.49|0.992|0.095|
||**BMGS**|**45.13**|**0.994**|**0.090**|
|RGB|w/o Calib|39.64|0.985|0.069|
||**BMGS**|**40.40**|**0.988**|**0.065**|

# Response to Reviewer kcrz
Thank you very much for your insightful comments on the novelty of our method. We are glad to clarify the fundamental differences between our proposed BMGS and Thermal Gaussian (OMMG). In the revised manuscript, we will add references to Thermal Gaussian and briefly introduce the OMMG architecture in Section 2.2 (Thermal IR and Multi-modal).

**1. Different prerequisites for input data**

Similar to methods like ThermoNeRF[1], OMMG requires specialized devices (FILR series cameras) to collect aligned RGB-IR images for reconstruction, typically at low resolutions (max. 640×480). In contrast, BMGS places no special requirements on equipment and flexibly supports non-aligned images at significantly higher resolutions (e.g., RGB at 3840×2160 and IR at 640×512). Thus, BMGS addresses a more challenging and novel scenes.

**2. Different Modeling Methods**

In OMMG, Gaussian appearance attributes include modality-specific spherical harmonics but still share opacity, which could induce gradient competition or feature interference between modalities. However, BMGS decouples spherical harmonics and opacity by modality, avoiding mutual interference, ensuring stable optimization and expressive flexibility.

**3. Different Optimization Methods**

OMMG simultaneously updates Gaussian parameters for both modalities under the same viewpoint, with shared opacity values. This coupling forces the model to "compromise" when conflicts arise between modalities (e.g., thermal background vs. visible textures). BMGS separately optimizes modality-specific appearance attributes (color + opacity) at the same geometric location from different viewpoints, maintaining spatial coherence via geometric consistency, thus preventing appearance interference.

**4. Richer Viewpoint Inputs**

In OMMG, each RGB-IR image pair is treated as a single viewpoint during modeling. However, BMGS treats RGB and IR images as two independent viewpoints, significantly increasing the density of overall viewpoint coverage. This provides benefits in strengthening geometric constraints, filling occluded areas, and enhancing tolerance to modality differences.

[1] Hassan M, Forest F, Fink O, et al. ThermoNeRF: Multimodal Neural Radiance Fields for Thermal Novel View Synthesis[J]. CoRR, 2024.

# Response to Reviewer 2rjJ
We sincerely thank you for your in-depth understanding and professional grasp of technical details.

**1. Comparison with Traditional Stereo Calibration Methods**

Traditional calibration tools (MATLAB, OpenCV) assume identical resolutions and modalities, limiting their effectiveness in RGB-IR, cross-modal scenes. Our BMGS performs modality-independent calibration, uses checkerboard captures for cross-modal constraints, and solves rotation and translation precisely via Frobenius minimization and SVD orthogonalization, achieving sub-pixel accuracy (0.242 px reprojection error), significantly outperforming traditional methods.

**2. Unifying COLMAP Scale with Real-World Scale**

We calculate a scaling factor $s$ by comparing the distance $L_{colmap}$ of feature points (e.g., checkerboard corners or object edges) in the COLMAP scale with the distance $L_{real}$ in the real world, thereby achieving the unification of the COLMAP scale and the real scale. The process is mathematically formulated as follows:

$R_{colmap} =R_{calib},~T_{colmap} =sT_{calib},~s=L_{colmap}/L_{real}$

We will revise Fig. 2 and Section 3.1 accordingly to clarify this procedure.

**3. Clarification of Symbols**

Indeed, the $\alpha$ on line 440 denotes opacity. However, the symbol used on line 433 is not $\alpha$, but rather the modality-specific Gaussian appearance attribute $a$, as defined in Eq.(5). To avoid confusion, we will replace the attribute $a$ with the symbol $\phi$ in the revised manuscript.

**4. Clarification of Formulas**

Eq. (4) defines the complete set of attributes for each Gaussian in the Unified Gaussian Field, explicitly noting that the color is represented via spherical harmonics per modality, as stated clearly on line 441: "$c^m_i(u)$ denotes the color represented by spherical harmonics under modality $m$."

**5. Differences between PARID and Existing Datasets**

Existing datasets[1-4] typically involve modality-aligned, same-resolution and single-modality data. None adequately address the misaligned scenes central to our work. Our proposed dataset PARID uniquely supports misaligned RGB-IR training, with high-resolution data and thermal annotations, making it ideal for evaluating non-aligned, cross-modal and cross-resolution methods.
|Dataset|Alignment|Modality|Resolution RGB|Resolution IR|
|-|-|--|-|-|
|Thermal-NeRF[1]|-|IR|-|382x288|
|ThermoScenes[2]|Aligned|RGB&IR|480x640|480x640|
|IR-NeRF[3]|-|IR|-|640x512|
|ThermalGaussian[4]|Aligned|RGB&IR|640x480|640x480|
|**ours PARID**|**Non-Aligned**|**RGB&IR**|**1920x1080**|**1920x1080**|

**6. Additional Baseline Methods**

We carefully considered suggested comparison methods ([1-6]). However, due to fundamental differences in underlying assumptions or unavailability of open-source code, fair and rigorous comparisons are difficult to perform. Specifically, methods [2,4,6] depend on factory-aligned imaging devices; methods [4,5] rely on manually warping images or using calibration targets before training; and methods [1,3] have not provided publicly available code. In contrast, BMGS explicitly targets non-aligned RGB-thermal image pairs without any external assumptions or constraints, underscoring our method’s generality and robustness. We believe these methodological differences highlight complementary directions within multi-modal reconstruction research. In the revised manuscript, we will cite and clearly discuss these methods to clarify these distinctions.

**7. Fundamental Differences between BMGS and Thermal Gaussian (OMMG)**

- Inputs: BMGS supports higher-resolution, non-aligned RGB-IR; OMMG requires aligned, specialized cameras.

- Modeling: BMGS fully decouples appearance (opacity/color) per modality; OMMG shares opacity, risking interference.

- Optimization: BMGS separately optimizes modalities via distinct viewpoints to prevent conflicts, whereas OMMG jointly optimizes modalities from a single viewpoint.

- Viewpoints: BMGS independently treats RGB and IR inputs, enriching viewpoint coverage(Double of OMMG), geometry constraints, and tolerance to modality variations.

**8. Ablation Study of the Calibration Module**

To demonstrate the calibration module’s critical importance, we conducted ablation studies from two key aspects:

**(1) Reprojection Error**: Removing the calibration module drastically increases reprojection error to **~12 px**, indicating severe misalignment. Using BMGS calibration achieves a sub-pixel accuracy of **~0.21 px**.
|Method|||REproj|||
|-|-|-|-|-|-|
||pair1|pair2|pair3|pair4|Avg|
|w/o Calib|12.32|12.29|12.05|11.97|12.16|
|**BMGS**|**0.25**|**0.25**|**0.14**|**0.21**|**0.21**|

**(2) Reconstruction Performance**: Without calibration, the "apples" scene experiences significant performance degradation (IR: -1.64 dB, RGB: -0.76 dB PSNR), highlighting calibration’s necessity.
|Modality|Method|PSNR↑|SSIM↑|LPIPS↓|
|-|-|-|-|-|
|IR|w/o Calib|43.49|0.992|0.095|
||**BMGS**|**45.13**|**0.994**|**0.090**|
|RGB|w/o Calib|39.64|0.985|0.069|
||**BMGS**|**40.40**|**0.988**|**0.065**|

# Response to Reviewer 1GTm
Thank you for your comments. However, we must clarify several points:

**1. Comparison with Thermal-NeRF**

Your suggestion to compare with Thermal-NeRF is valid, but their code is currently unavailable. If we obtain the code, we will include this comparison.

**2. Sensitivity Analysis**

The number of Gaussian $M$ in BMGS is not a fixed hyperparameter but dynamically adjusted by 3DGS pruning and densification strategies. Table below shows analysis of spherical harmonics degree $D$. Clearly, BMGS performance is robust to $D$, with $D=3$ optimal.
|Modality|Metric||D||
|-|-|-|-|-|
|||1|2|3|
|IR|PSNR↑|41.17|41.65|41.72|
||SSIM↑|0.977|0.977|0.978|
||LPIPS↓|0.232|0.231|0.230|
|RGB|PSNR↑|31.59|31.77|32.15|
||SSIM↑|0.962|0.964|0.966|
||LPIPS↓|0.064|0.051|0.058|

**3. Runtime Analysis**

Calibration, training, and inference stages are independent; thus, an end-to-end runtime isn't meaningful. For the "desktop" scene: Calibration ~30s, Training ~32 it/s (converging ~15-20min), Rendering (RGB 210 FPS, IR 261 FPS).

**4. Regarding Temporal Consistency**

Our method explicitly targets static scenes. Systematic dynamic evaluation is beyond this paper’s scope. Future work will consider explicit temporal modeling.

**5. Incremental Updates**

You ask about dynamic Gaussian updates, clearly ignoring our paper’s explicit focus on static scenes. Incremental updates are relevant to dynamic reconstruction, entirely beyond our stated research scope. We will note this as future work, but it does not pertain to our current contributions.

**6. Robustness of Cross-Modal Calibration**

- Significant FOV Difference: BMGS effectively manages substantial FOV differences between RGB and IR cameras. The table below clearly shows these FOV discrepancies, demonstrating that BMGS's geometric consistency constraints can robustly handle significant viewpoint variations.

|Camera|FOVx(°)|FOVy(°)|
|-|-|-|
|RGB|50.74|29.86|
|IR|45.14|36.80|

- Non-Rigid Deformations: Currently, BMGS uses rigid poses obtained via COLMAP, which itself does not handle non-rigid deformations. In future work, we plan to integrate a joint calibration-reconstruction optimization strategy, allowing small adjustments to camera extrinsics during training. This approach aims to mitigate errors from vibrations or mounting deformations.
