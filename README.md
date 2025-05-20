# AI Paper Digest

- Super-brief summaries of AI papers I've read
- May not perfectly align with authors' claims or intentions
- Some papers I think important include detailed summary links, which leads to my blog posts


---


> **Vision-Language Models Do Not Understand Negation**  
> *CVPR 2025*, [arxiv](https://arxiv.org/abs/2501.09425), [website](https://negbench.github.io/), [code](https://github.com/m1k2zoo/negbench)  
> Task: understanding how well VLMs handle negation  
> 
> - How well do current VLMs understand negation?  
> - To comprehensively evaluate how well VLMs handle negation, propose NegBench.  
> - Joint embedding-based VLMs, such as CLIP, frequently collapse affirmative and negated statements into similar embeddings.  
> - Data-centric approach is effective: fine-tuning CLIP-based models on large-scale datasets containing millions of negated captions.  



> **ArtiFade: Learning to Generate High-quality Subject from Blemished Images**  
> *CVPR 2025*, [arxiv](https://arxiv.org/abs/2409.03745)  
> Task: blemished subject-driven generation  
> 
> - Current subject-driven T2I methods are vulnerable to artifacts such as watermarks, stickers, or adversarial noise.  
> - This limitation arises because current methods lack the discriminative power to distinguish subject-related features from disruptive artifacts.  
> - Propose ArtiFade, which generates high-quality artifact-free images from blemished datasets.  
> - Core method: artifact rectification training, which first reconstructs the blemished image and then learns to rectify it into an unblemished version.  



> **Detecting, Explaining, and Mitigating Memorization in Diffusion Models**  
> *ICLR 2024 Oral*, [arxiv](https://arxiv.org/abs/2407.21720), [review](https://openreview.net/forum?id=84n3UwkH7b), [code](https://github.com/YuxinWenRick/diffusion_memorization)  
> Task: detect and mitigate memorization in diffusion models  
> 
> - For memorized prompts, the text condition consistently guides the generation towards the memorized solution, regardless of the initializations.  
> - Thus, memorized prompts tend to exhibit larger magnitudes than non-memorized ones.  
> - Detect memorization: $$\epsilon_{\theta}(x_t, e_{prompt}) - \epsilon_{\theta}(x_t, e_{\phi})$$.  
> - Detect trigger tokens: $$\nabla_{e^{token}} \left\| \epsilon_{\theta}(x_t, e_{prompt}) - \epsilon_{\theta}(x_t, e_{\phi}) \right\|$$.  
> - Inference-time mitigation: minimize the magnitude of text-conditional noise prediction at the initial timestep.  
> - Training-time mitigation: exclude samples if the magnitude of text-conditional noise prediction surpasses a predetermined threshold.  



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



> **AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection**  
> *ICLR 2024*, [arxiv](https://arxiv.org/abs/2310.18961), [review](https://openreview.net/forum?id=buC4E91xZE), [code](https://github.com/zqhang/AnomalyCLIP), [summary](https://jasonleex1995.github.io/docs/07_papers/2310.18961.html)  
> Task: zero-shot anomaly detection (ZSAD)  
> 
> - Previous works use CLIP with object-aware text prompts.  
> - Even though the foreground object semantics can be completely different, anomaly patterns remain quite similar.  
> - Thus, use CLIP with learnable object-agnostic text prompts.  



> **Tiny and Efficient Model for the Edge Detection Generalization**  
> *ICCV 2023 Workshop (Resource Efficient Deep Learning for Computer Vision)*, [arxiv](https://arxiv.org/abs/2308.06468), [code](https://github.com/xavysp/TEED)   
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
> 
> - T2I models struggle with fine-grained spatial control.  
> - Incorporate additional images as conditioning inputs to generate images with spatial control.  
> - Fine-tuning a trainable copy of a T2I diffusion model connected with zero convolution.  



> **Learning Universal Policies via Text-Guided Video Generation**  
> *NeurIPS 2023 Spotlight*, [arxiv](https://arxiv.org/abs/2302.00111), [review](https://openreview.net/forum?id=bo8q5MRcwy), [website](https://universal-policy.github.io/unipi/)  
> Task: robot manipulation  
> 
> - Challenges of training robots: environmental diversity due to joint modeling of space & controls.  
> - Propose Universal Policy (UniPi): cast the sequential decision making problem as a text-conditioned video generation problem.  
> - Plans actions by generating a video for a desired goal, then infers control actions using an inverse dynamics model.  



> **DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation**  
> *CVPR 2023 Award Candidate*, [arxiv](https://arxiv.org/abs/2208.12242), [website](https://dreambooth.github.io/), [code](https://github.com/google/dreambooth), [summary](https://jasonleex1995.github.io/docs/07_papers/2208.12242.html)  
> Task: subject-driven image generation  
> 
> - Recently developed large T2I diffusion models can generate high-quality and diverse photorealistic images.  
> - However, these models lack the ability to mimic the appearance of subjects in a given reference set.  
> - Generate novel photorealistic images of the subject contextualized in different scenes via fine-tuning with rare tokens and class-specific prior preservation loss.  



> **Prompt-to-Prompt Image Editing with Cross Attention Control**  
> *ICLR 2023 Spotlight*, [arxiv](https://arxiv.org/abs/2208.01626), [review](https://openreview.net/forum?id=_CDixzkzeyb), [website](https://prompt-to-prompt.github.io/), [code](https://github.com/google/prompt-to-prompt/), [summary](https://jasonleex1995.github.io/docs/07_papers/2208.01626.html)  
> Task: text-driven image editing  
> 
> - Previous works use spatial mask to edit an image.  
> - These methods generate unsatisfactory results caused by ignoring the original structure and content within the masked region.  
> - Propose Prompt-to-Propmt, a text-driven image editing by injecting the cross-attention maps of original prompt to the cross-attention maps of edited prompt.  



> **An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion**  
> *ICLR 2023 Spotlight*, [arxiv](https://arxiv.org/abs/2208.01618), [review](https://openreview.net/forum?id=NAQvF08TcyG), [website](https://textual-inversion.github.io/), [code](https://github.com/rinongal/textual_inversion), [summary](https://jasonleex1995.github.io/docs/07_papers/2208.01618.html)  
> Task: personalized text-to-image generation  
> 
> - Recently, large-scale T2I models have demonstrated an unprecedented capability to reason over natural language descriptions.  
> - However, generating a desired target, such as user-specific concept, through text is quite difficult.  
> - Training T2I models have several limitations.  
> - Generate novel photorealistic images of the subject via optimizing only a single word embedding.  



> **Do As I Can, Not As I Say: Grounding Language in Robotic Affordances**  
> *CoRL 2022 Oral*, [arxiv](https://arxiv.org/abs/2204.01691), [review](https://openreview.net/forum?id=bdHkMjBJG_w), [website](https://say-can.github.io/), [code](https://github.com/google-research/google-research/tree/master/saycan)  
> Task: robot manipulation and navigation  
> 
> - How can we enable robots to perform complex tasks in the real world?  
> - Propose SayCan: given pre-trained low-level skills, an LLM (Say) determines useful actions for a high-level goal, and a learned affordance function (Can) determines whether action is possible to execute in the current state.  
> - Only train affordance function using reinforcement learning (RL).  



> **Learning to generate line drawings that convey geometry and semantics**  
> *CVPR 2022*, [arxiv](https://arxiv.org/abs/2203.12691), [website](https://carolineec.github.io/informative_drawings/), [code](https://github.com/carolineec/informative-drawings)  
> Task: automatic line generation  
> 
> - View line drawing generation as an unsupervised image translation problem, which means training models with unpaired data.  
> - Previous works solely consider preserving photographic appearence through cycle consistency.  
> - Instead, use 4 losses to improve quality: adversarial loss (LSGAN), geometry loss (pseudo depth map), semantic loss (CLIP), appearance loss (cycle consistency).  



> **SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations**  
> *ICLR 2022*, [arxiv](https://arxiv.org/abs/2108.01073), [review](https://openreview.net/forum?id=aBsCjcPu_tE), [website](https://sde-image-editing.github.io/), [code](https://github.com/ermongroup/SDEdit), [summary](https://jasonleex1995.github.io/docs/07_papers/2108.01073.html)  
> Task: guided image synthesis & editing  
> 
> - Previous works on guided image synthesis or editing: conditional GANs or GAN inversions  
> - Conditional GANs require collecting training images and training per task.  
> - GAN inversions require task-specific loss design.  
> - Generate realistic images by adding small noise and denoising with score-based models trained on the target domain.  



> **Generalisation in humans and deep neural networks**    
> *NeurIPS 2018*, [arxiv](https://arxiv.org/abs/1808.08750), [review](https://papers.nips.cc/paper_files/paper/2018/hash/0937fb5864ed06ffb59ae5f9b5ed67a9-Abstract.html), [code](https://github.com/rgeirhos/generalisation-humans-DNNs), [summary](https://jasonleex1995.github.io/docs/07_papers/1808.08750.html)  
> Task: understanding the differences between DNNs and humans   
> 
> - Compare the robustness of humans and DNNs (VGG, GoogLeNet, ResNet) on object recognition under 12 different image distortions.  
> - Human visual system is more robust than DNNs.  
> - DNNs generalize so poorly under non-i.i.d. settings.  



---
```
format

> **paper title**  
> *accept info*, [arxiv](), [review](), [website](), [code](), [summary]()  
> Task:  
> 
> - super-brief summary

```