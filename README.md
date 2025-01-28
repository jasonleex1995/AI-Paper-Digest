# AI Paper Digest

- Super-brief summaries of AI papers I've read
- May not perfectly align with authors' claims or intentions
- Some papers I think important include detailed summary links, which leads to my blog posts


---



**AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection**  
*ICLR 2024*, [arxiv](https://arxiv.org/abs/2310.18961), [review](https://openreview.net/forum?id=buC4E91xZE), [code](https://github.com/zqhang/AnomalyCLIP), [summary](https://jasonleex1995.github.io/docs/07_papers/2310.18961.html)  
Task: zero-shot anomaly detection (ZSAD)  
Previous works use CLIP with object-aware text prompts.  
Even though the foreground object semantics can be completely different, anomaly patterns remain quite similar.  
Thus, learning object-agnostic text prompts is the key for ZSAD.  



**Learning to generate line drawings that convey geometry and semantics**  
*CVPR 2022*, [arxiv](https://arxiv.org/abs/2203.12691), [code](https://github.com/carolineec/informative-drawings), [website](https://carolineec.github.io/informative_drawings/)  
Task: automatic line generation  
View line drawing generation as an unsupervised image translation problem, which means training models with unpaired data.  
Most previous works solely consider preserving photographic appearence through cycle consistency.  
Instead, use 4 losses to improve quality: adversarial loss (LSGAN), geometry loss (pseudo depth map), semantic loss (CLIP), appearance loss (cycle consistency).  



**Generalisation in humans and deep neural networks**    
*NeurIPS 2018*, 
[arxiv](https://arxiv.org/abs/1808.08750), [review](https://papers.nips.cc/paper_files/paper/2018/hash/0937fb5864ed06ffb59ae5f9b5ed67a9-Abstract.html), [code](https://github.com/rgeirhos/generalisation-humans-DNNs), [summary](https://jasonleex1995.github.io/docs/07_papers/1808.08750.html)  
Compare the robustness of humans and DNNs (VGG, GoogLeNet, ResNet) on object recognition under 12 different image distortions.  
Human visual system is more robust than DNNs.  
DNNs generalize so poorly under non-i.i.d. settings.  



---
```
format

**paper title**  
*accept info*, [arxiv](), [review](), [code](), [website](), [summary]()  
super-brief summary

```