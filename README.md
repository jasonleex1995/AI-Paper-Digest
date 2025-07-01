# AI Paper Digest

- Super-brief summaries of AI papers I've read
- May not perfectly align with authors' claims or intentions
- Some papers I think important include detailed summary links, which leads to my blog posts


---



> **Augmenting Perceptual Super-Resolution via Image Quality Predictors**  
> *CVPR 2025*, [arxiv](https://arxiv.org/abs/2504.18524)  
> Task: single image super-resolution  
> 
> - Previous work shows that human feedback (manual human ranking) can improve SR performance.  
> - However, this approach is not scalable and coarse.  
> - Use NR-IQA metric (MUSIQ) to replace human feedback.  
> - Improve perceptual quality using highest-score patch selection & direct optimization.  



> **Dual Prompting Image Restoration with Diffusion Transformers**  
> *CVPR 2025*, [arxiv](https://arxiv.org/abs/2504.17825)  
> Task: image restoration  
> 
> - Diffusion Transformers (DiTs) recently show promising generative capabilities.  
> - However, effectively incorporating low-quality (LQ) image information into DiTs remains underexplored.  
> - Propose Dual Prompting Image Restoration (DPIR), which effectively integrate control signals from LQ images into the DiT.  
> - 3 key components: degradation-robust VAE encoder, low-quality image conditioning branch, dual prompting control branch.  
> - Low-quality image conditioning: use VAE latent of LQ image for conditioning.  
> - Dual prompting control: use VAE output of LQ image for prompt instead of text.  



> **Acquire and then Adapt: Squeezing out Text-to-Image Model for Image Restoration**  
> *CVPR 2025*, [arxiv](https://arxiv.org/abs/2504.15159)  
> Task: image restoration  
> 
> - Pre-trained T2I model based restoration approach can enrich low-quality images of any type with realistic details, but requires immense dataset and extensive training to prevent alterations of image content.  
> - Propose FluxGen: generate diverse realistic high-quality images using Flux with empty prompt, and then filter using NR-IQA metrics.  
> - Propose FluxIR: light-weighted ControlNet-like adapter, which controls all the MM-DiT block of Flux model using squeeze-and-excitation (SE) layers.  
> - Train FluxIR with FluxGen with modified timestep sampling & additional pixel-space loss.  



> **Progressive Focused Transformer for Single Image Super-Resolution**  
> *CVPR 2025*, [arxiv](https://arxiv.org/abs/2503.20337), [code](https://github.com/LabShuHangGU/PFT-SR)  
> Task: single-image super-resolution  
> 
> - Previous Transformer-based SR methods use vanilla or sparse self-attention, which still compute attention between query tokens and irrelevant tokens, leading to unnecessary computations.  
> - Intuition: highly relevant tokens will be consistently similar to each other across layers, so use previous layer's attention maps to identify relevant tokens.  
> - Propose Progressive Focused Attention (PFA): calculate current layer's PFA maps by the Hadamard product of the previous layer's PFA maps and the current layer's self-attention map, followed by top-k selection to construct sparse attention maps.  



> **Exploring Semantic Feature Discrimination for Perceptual Image Super-Resolution and Opinion-Unaware No-Reference Image Quality Assessment**  
> *CVPR 2025*, [arxiv](https://arxiv.org/abs/2503.19295), [code](https://github.com/GuangluDong0728/SFD)  
> Task: single-image super-resolution  
> 
> - Previous GAN-based SR methods directly discriminate on images without semantic awareness, causing the generated textures to misalign with image semantics.  
> - Propose Semantic Feature Discrimination (SFD): discriminate multi-scale CLIP semantic features.  
> - Since previous discriminator-based opinion-unaware no-reference image quality assessment (OU NR-IQA) methods ignore the assessment for semantic, the discriminator trained with SFD achieves better performance on OU NR-IQA.  



> **Uncertainty-guided Perturbation for Image Super-Resolution Diffusion Model**  
> *CVPR 2025*, [arxiv](https://arxiv.org/abs/2503.18512), [code](https://github.com/LabShuHangGU/UPSR)  
> Task: single image super-resolution  
> 
> - Previous diffusion-based SR methods inject a large initial noise into the LR image, which is not specialized and effective for SR.  
> - Four core methods: Uncertainty-guided Noise Weighting (UNW), additional SR image conditioning, loss design, and network architecture design.  
> - UNW: low noise in flat areas, large noise in edge and texture areas.  
> - SR image conditioning: combine the pre-trained SR model output to provide more accurate conditional information.  
> - Loss design: fidelity loss (RMSE) + perceptual loss (LPIPS)  
> - Network archiecture design: encoder & decoder → PixelUnshuffle & nearest neighbor sampling.  



> **The Power of Context: How Multimodality Improves Image Super-Resolution**  
> *CVPR 2025*, [arxiv](https://arxiv.org/abs/2503.14503)  
> Task: single image super-resolution  
> 
> - Previous text-prompt-driven SR methods use captions generated by VLMs, but VLMs cannot accurately represent spatial information, leading to hallucinated details.  
> - Propose Multimodal Super-Resolution (MMSR): incorporate additional spatial modalities (depth, segmentation, edge) into a diffusion model to implicitly align language descriptions with spatial regions.  
> - Instead of using ControlNet-style conditioning, use pretrained VQGAN image tokenizer to encode diverse modalities into a unified token representation, which are then concatenated with text embeddings and used in cross-attention within the diffusion model.  



> **Vision-Language Models Do Not Understand Negation**  
> *CVPR 2025*, [arxiv](https://arxiv.org/abs/2501.09425), [website](https://negbench.github.io/), [code](https://github.com/m1k2zoo/negbench)  
> Task: understanding how well VLMs handle negation  
> 
> - How well do current VLMs understand negation?  
> - To comprehensively evaluate how well VLMs handle negation, propose NegBench.  
> - Joint embedding-based VLMs, such as CLIP, frequently collapse affirmative and negated statements into similar embeddings.  
> - Data-centric approach is effective: fine-tuning CLIP-based models on large-scale datasets containing millions of negated captions.  


> **Generative Photography: Scene-Consistent Camera Control for Realistic Text-to-Image Synthesis**  
> *CVPR 2025 Highlight*, [arxiv](https://arxiv.org/abs/2412.02168), [website](https://generative-photography.github.io/project/), [code](https://github.com/pandayuanyu/generative-photography)  
> Task: generative photography  
> Scene-consistent text-to-image generation with camera intrinsics control by fine-tuning a T2V with a differential camera encoder.



> **Classifier-Free Guidance inside the Attraction Basin May Cause Memorization**  
> *CVPR 2025*, [arxiv](https://arxiv.org/abs/2411.16738), [code](https://github.com/anubhav1997/mitigating_memorization)  
> Task: mitigate memorization in diffusion models  
> 
> - Applying classifier-free guidance (CFG) before a certain timestep (transition point) tends to produce memorized samples.  
> - Applying CFG after the transition point is unlikely to yield a memorized image.  
> - Although every prompt and initialization pair leads to a different transition point, the transition point can be found by identifying the first local minima of the graph of $$\epsilon_{\theta}(x_t, e_{prompt}) - \epsilon_{\theta}(x_t, e_{\phi})$$.  
> - Propose Opposite Guidance (OG): apply opposite CFG until the transition point, and switch to traditional positive CFG after the transition point.  



> **ArtiFade: Learning to Generate High-quality Subject from Blemished Images**  
> *CVPR 2025*, [arxiv](https://arxiv.org/abs/2409.03745)  
> Task: blemished subject-driven generation  
> 
> - Current subject-driven T2I methods are vulnerable to artifacts such as watermarks, stickers, or adversarial noise.  
> - This limitation arises because current methods lack the discriminative power to distinguish subject-related features from disruptive artifacts.  
> - Propose ArtiFade, which generates high-quality artifact-free images from blemished datasets.  
> - Core method: artifact rectification training, which first reconstructs the blemished image and then learns to rectify it into an unblemished version.  



> **Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model**  
> *arXiv 2024*, [arxiv](https://arxiv.org/abs/2408.11039)  
> Task: multi-modal generation  
> 
> - Previous works quantize continuous modalities and train with next-token prediction.  
> - Propose Transfusion: generate discrete and continuous modalities using a different objective for each modality.  
> - Use next token prediction for text and diffuion for images.  
> - For images, use unrestricted (bidirectional) attention.  



> **Detecting, Explaining, and Mitigating Memorization in Diffusion Models**  
> *ICLR 2024 Oral*, [arxiv](https://arxiv.org/abs/2407.21720), [review](https://openreview.net/forum?id=84n3UwkH7b), [code](https://github.com/YuxinWenRick/diffusion_memorization)  
> Task: detect and mitigate memorization in diffusion models  
> 
> - For memorized prompts, the text condition consistently guides the generation towards the memorized solution, regardless of the initializations.  
> - Thus, memorized prompts tend to exhibit larger magnitudes than non-memorized ones.  
> - Use $$\epsilon_{\theta}(x_t, e_{prompt}) - \epsilon_{\theta}(x_t, e_{\phi})$$ to detect and mitigate memorization.  
> - Detect trigger tokens by measure the influence of $$\epsilon_{\theta}(x_t, e_{prompt}) - \epsilon_{\theta}(x_t, e_{\phi})$$ per each token.  



> **Identity Decoupling for Multi-Subject Personalization of Text-to-Image Models**  
> *NeurIPS 2024*, [arxiv](https://arxiv.org/abs/2404.04243), [review](https://openreview.net/forum?id=tEEpVPDaRf), [website](https://mudi-t2i.github.io/), [code](https://github.com/agwmon/MuDI)  
> Task: multi-subject personalization  
> 
> - Previous works on personalization suffer from identity mixing when composing multiple subjects.  
> - During training, use detailed descriptions and Seg-Mix augmentation, which randomly composes segmented subjects.  
> - During inference, use mean-shifted noise instead of Gaussian noise, which use the segmented subjects to initialize.  
> - Propose new metric Detect-and-Compare (D&C) to evaluate multi-subject fidelity.  



> **Ferret-UI: Grounded Mobile UI Understanding with Multimodal LLMs**  
> *ECCV 2024*, [arxiv](https://arxiv.org/abs/2404.05719), [code](https://github.com/apple/ml-ferret)  
> Task: referring, grounding, and reasoning on mobile UI screens  
> 
> - Directly adapting MLLMs to UI screens has limitation, since UI screens exhibit more elongated aspect ratios and contain smaller objects of interests than natural images.  
> - Incorporate "any resolution" (anyres) on top of Ferret, and then train with curated dataset.  
> - During training, both the decoder and the projection layer are updated while the vision encoder is kept frozen.  



> **Unveiling and Mitigating Memorization in Text-to-image Diffusion Models through Cross Attention**  
> *ECCV 2024*, [arxiv](https://arxiv.org/abs/2403.11052), [code](https://github.com/renjie3/MemAttn)  
> Task: detect and mitigate memorization in text-to-image diffusion models  
> 
> - Since memorized images are usually triggered by the memorized text prompts, cross attention can exhibit unique behaviors specific to memorization.  
> - Memorized samples tend to allocate most of the attention to specific tokens throughout all the diffusion steps, where non-memorized samples have more dispersed attention distribution.  
> - Thus, use entropy of attention to detect and mitigate memorization.  



> **AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection**  
> *ICLR 2024*, [arxiv](https://arxiv.org/abs/2310.18961), [review](https://openreview.net/forum?id=buC4E91xZE), [code](https://github.com/zqhang/AnomalyCLIP), [summary](https://jasonleex1995.github.io/docs/07_papers/2310.18961.html)  
> Task: zero-shot anomaly detection (ZSAD)  
> 
> - Previous works use CLIP with object-aware text prompts.  
> - Even though the foreground object semantics can be completely different, anomaly patterns remain quite similar.  
> - Thus, use CLIP with learnable object-agnostic text prompts.  



> **Tiny and Efficient Model for the Edge Detection Generalization**  
> *ICCV 2023 Workshop*, [arxiv](https://arxiv.org/abs/2308.06468), [code](https://github.com/xavysp/TEED)   
> Task: edge detection  
> 
> - Propose simple, efficient, and robust CNN model: Tiny and Efficient Edge Detector (TEED).  
> - TEED generates thinner and clearer edge-maps, but requires a paired dataset for training.  
> - Two core methods: architecture (edge fusion module) & loss (weighted cross-entropy, tracing loss).  
> - Weighted cross-entropy helps to detect as many edges as possible, while tracing loss helps to predict thinner and clearer edge-maps.  



> **RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control**  
> *CoRL 2023*, [arxiv](https://arxiv.org/abs/2307.15818), [review](https://openreview.net/forum?id=XMQgwiJ7KSX), [website](https://robotics-transformer2.github.io/)  
> Task: robot manipulation  
> 
> - Previous works primarily used LLMs and VLMs for high-level robot planning.  
> - Propose RT-2, which directly integrates large pre-trained VLMs into low-level robot control.
> - Two core methods: tokenizing the actions into text tokens & co-fine-tuning robotics data with the original web data.  



> **Understanding and Mitigating Copying in Diffusion Models**  
> *NeurIPS 2023*, [arxiv](https://arxiv.org/abs/2305.20086), [review](https://openreview.net/forum?id=HtMXRGbUMt), [code](https://github.com/somepago/DCR)  
> Task: analyze and mitigate memorization in T2I diffusion models  
> 
> - Previous works show that duplicate images in the training set cause memorization in diffusion models.  
> - Reveals that text conditioning plays a major role in memorization.  
> - Inference-time mitigation: use random token replacement or addition strategy, where random tokens are randomly added or replaced into the text prompt.  
> - Train-time mitigation: use multiple caption strategy, where each image has diverse captions and caption is randomly sampled during training.  



> **Implicit Diffusion Models for Continuous Super-Resolution**  
> *CVPR 2023*, [arxiv](https://arxiv.org/abs/2303.16491), [code](https://github.com/Ree1s/IDM)  
> Task: single-image super-resolution  
> Continuous image super-resolution by replacing the U-Net decoder with an implicit neural representation and conditioning on multi-resolution LR features.



> **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion**  
> *RSS 2023*, [arxiv](https://arxiv.org/abs/2303.04137), [website](https://diffusion-policy.cs.columbia.edu/), [code](https://github.com/real-stanford/diffusion_policy)  
> Task: robot manipulation via behavior cloning  
> 
> - Due to unique nature of predicting robot actions (ex. multimodal action distributions, sequential correlation, requirement of high precision), policy learning from demonstration is more challenging than other supervised learning problems.  
> - Diffusion models can represent multimodal distributions, scale to high-dimension output spaces, and achieve stable training while maintaining expressivity.  
> - Propose Diffusion policy, which is conditioned on visual representation and predicts high-dimensional action sequences.



> **Adding Conditional Control to Text-to-Image Diffusion Models**  
> *ICCV 2023 Oral*, [arxiv](https://arxiv.org/abs/2302.05543), [code](https://github.com/lllyasviel/ControlNet), [summary](https://jasonleex1995.github.io/docs/07_papers/2302.05543.html)  
> Task: image-based conditional image generation  
> Fine-tune a trainable copy of a T2I diffusion model, connected via zero convolution, to achieve fine-grained spatial control using additional images as conditioning inputs.


> **Learning Universal Policies via Text-Guided Video Generation**  
> *NeurIPS 2023 Spotlight*, [arxiv](https://arxiv.org/abs/2302.00111), [review](https://openreview.net/forum?id=bo8q5MRcwy), [website](https://universal-policy.github.io/unipi/)  
> Task: robot manipulation  
> Plans actions by generating a goal-directed video using a T2V diffusion model, and then infers control actions from the video using an inverse dynamics model. 


> **DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation**  
> *CVPR 2023 Award Candidate*, [arxiv](https://arxiv.org/abs/2208.12242), [website](https://dreambooth.github.io/), [code](https://github.com/google/dreambooth), [summary](https://jasonleex1995.github.io/docs/07_papers/2208.12242.html)  
> Task: subject-driven image generation  
> Generate novel photorealistic images of the subject contextualized in different scenes via fine-tuning with rare tokens and class-specific prior preservation loss.


> **Prompt-to-Prompt Image Editing with Cross Attention Control**  
> *ICLR 2023 Spotlight*, [arxiv](https://arxiv.org/abs/2208.01626), [review](https://openreview.net/forum?id=_CDixzkzeyb), [website](https://prompt-to-prompt.github.io/), [code](https://github.com/google/prompt-to-prompt/), [summary](https://jasonleex1995.github.io/docs/07_papers/2208.01626.html)  
> Task: text-driven image editing  
> Text-driven image editing by injecting the cross-attention maps of original prompt to the cross-attention maps of edited prompt.


> **An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion**  
> *ICLR 2023 Spotlight*, [arxiv](https://arxiv.org/abs/2208.01618), [review](https://openreview.net/forum?id=NAQvF08TcyG), [website](https://textual-inversion.github.io/), [code](https://github.com/rinongal/textual_inversion), [summary](https://jasonleex1995.github.io/docs/07_papers/2208.01618.html)  
> Task: personalized text-to-image generation  
> Generate novel photorealistic images of the subject via optimizing only a single word embedding.


> **Do As I Can, Not As I Say: Grounding Language in Robotic Affordances**  
> *CoRL 2022 Oral*, [arxiv](https://arxiv.org/abs/2204.01691), [review](https://openreview.net/forum?id=bdHkMjBJG_w), [website](https://say-can.github.io/), [code](https://github.com/google-research/google-research/tree/master/saycan)  
> Task: robot manipulation and navigation  
> Enable robots to perform complex real-world tasks by selecting appropriate low-level skills through high-level planning using LLM + affordance model.


> **MuLUT: Cooperating Multiple Look-Up Tables for Efficient Image Super-Resolution**  
> *ECCV 2022*, [paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1756_ECCV_2022_paper.php), [website](https://mulut.pages.dev/), [code](https://github.com/ddlee-cn/MuLUT)  
> Task: single-image super-resolution  
> Increase receptive field size of LUT efficiently by using complementary indexing (parallel), hierarchical indexing (cascade), and fine-tuning interpolation values.


> **Learning to generate line drawings that convey geometry and semantics**  
> *CVPR 2022*, [arxiv](https://arxiv.org/abs/2203.12691), [website](https://carolineec.github.io/informative_drawings/), [code](https://github.com/carolineec/informative-drawings)  
> Task: automatic line generation  
> Line drawing via unpaired image-to-image translation with 4 losses: adversarial loss (LSGAN), geometry loss (pseudo depth map), semantic loss (CLIP), appearance loss (cycle consistency).


> **SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations**  
> *ICLR 2022*, [arxiv](https://arxiv.org/abs/2108.01073), [review](https://openreview.net/forum?id=aBsCjcPu_tE), [website](https://sde-image-editing.github.io/), [code](https://github.com/ermongroup/SDEdit), [summary](https://jasonleex1995.github.io/docs/07_papers/2108.01073.html)  
> Task: guided image synthesis & editing  
> Generate realistic images by adding small noise and denoising with score-based models trained on the target domain.


> **Practical Single-Image Super-Resolution Using Look-Up Table**  
> *CVPR 2021*, [paper](https://openaccess.thecvf.com/content/CVPR2021/html/Jo_Practical_Single-Image_Super-Resolution_Using_Look-Up_Table_CVPR_2021_paper.html), [code](https://github.com/yhjo09/SR-LUT)  
> Task: single-image super-resolution  
> Practical SR by approximating small receptive field SR model into LUT, achieving similar runtime but better performance compared to interpolation methods.


> **Which Tasks Should Be Learned Together in Multi-task Learning?**  
> *ICML 2020*, [arxiv](https://arxiv.org/abs/1905.07553), [review](https://openreview.net/forum?id=HJlTpCEKvS), [website](http://taskgrouping.stanford.edu/), [code](https://github.com/tstandley/taskgrouping)  
> Task: multi-task learning  
> Many common assumptions do not seem to be true: more similar tasks don't necessarily work better together & task relationships are sensitive to dataset size and network capacity.


> **Generalisation in humans and deep neural networks**    
> *NeurIPS 2018*, [arxiv](https://arxiv.org/abs/1808.08750), [review](https://papers.nips.cc/paper_files/paper/2018/hash/0937fb5864ed06ffb59ae5f9b5ed67a9-Abstract.html), [code](https://github.com/rgeirhos/generalisation-humans-DNNs), [summary](https://jasonleex1995.github.io/docs/07_papers/1808.08750.html)  
> Task: understanding the differences between DNNs and humans   
> Compared to human visual system, DNNs (VGG, GoogLeNet, ResNet) generalize so poorly under non-i.i.d. settings.


> **Enhanced Deep Residual Networks for Single Image Super-Resolution**  
> *CVPR 2017 Workshop*, [arxiv](https://arxiv.org/abs/1707.02921), [code](https://github.com/limbee/NTIRE2017)  
> Task: single-image super-resolution  
> Optimize network and training for SR: remove batch normalization layer, train with residual scaling and L1 loss.



---
```
format

> **paper title**  
> *accept info*, [arxiv](), [review](), [website](), [code](), [summary]()  
> Task:  
> super-brief summary

```