# AI Paper Digest

- Super-brief summaries of AI papers I've read
- May not perfectly align with authors' claims or intentions
- Some papers I think important include detailed summary links, which leads to my blog posts


---



> **Ferret-UI: Grounded Mobile UI Understanding with Multimodal LLMs**  
> *ECCV 2024*, [arxiv](https://arxiv.org/abs/2404.05719), [code](https://github.com/apple/ml-ferret)  
> Task: referring, grounding, and reasoning on mobile UI screens  
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
> Propose simple, efficient, and robust CNN model: Tiny and Efficient Edge Detector (TEED).  
> TEED generates thinner and clearer edge-maps, but requires a paired dataset for training.  
> Two core methods: architecture (edge fusion module) & loss (weighted cross-entropy, tracing loss).  
> Weighted cross-entropy helps to detect as many edges as possible, while tracing loss helps to predict thinner and clearer edge-maps.  



> **Learning to generate line drawings that convey geometry and semantics**  
> *CVPR 2022*, [arxiv](https://arxiv.org/abs/2203.12691), [code](https://github.com/carolineec/informative-drawings), [website](https://carolineec.github.io/informative_drawings/)  
> Task: automatic line generation  
> 
> View line drawing generation as an unsupervised image translation problem, which means training models with unpaired data.  
> Most previous works solely consider preserving photographic appearence through cycle consistency.  
> Instead, use 4 losses to improve quality: adversarial loss (LSGAN), geometry loss (pseudo depth map), semantic loss (CLIP), appearance loss (cycle consistency).  



> **Generalisation in humans and deep neural networks**    
> *NeurIPS 2018*, [arxiv](https://arxiv.org/abs/1808.08750), [review](https://papers.nips.cc/paper_files/paper/2018/hash/0937fb5864ed06ffb59ae5f9b5ed67a9-Abstract.html), [code](https://github.com/rgeirhos/generalisation-humans-DNNs), [summary](https://jasonleex1995.github.io/docs/07_papers/1808.08750.html)  
> Task: understanding the differences between DNNs and humans   
> 
> Compare the robustness of humans and DNNs (VGG, GoogLeNet, ResNet) on object recognition under 12 different image distortions.  
> Human visual system is more robust than DNNs.  
> DNNs generalize so poorly under non-i.i.d. settings.  



---
```
format

> **paper title**  
> *accept info*, [arxiv](), [review](), [code](), [website](), [summary]()  
> Task:  
> 
> super-brief summary

```