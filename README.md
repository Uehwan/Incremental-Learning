# Incremental Learning
Incremental Learning with Adaptive Resonance Theory (ART) & Developmental Resonance networks

## 1. Online Incremental Learning
Online incremental learning without forgetting aims to analyze sequentially incoming data.
Online incremental learning allows robots to deal with dynamic environment information.
We propose s-DRN which can cluster sequential data with the following featuers:

- Online incremental: s-DRN processes input data and generates clusters dynamically.
- Computationally efficient: s-DRN requires O(n) computation.
- Robust to hyper-parameter setting: the performance of s-DRN is hardly affected by the internal hyper-parameters such as vigilance parameters.

For detailed formulation of s-DRN, please refer to [the sDRN directory](sDRN) of our repository and the following paper:

```
@article{yoon2019stabilized,
  title={Stabilized Developmental Resonance Network},
  author={Inug Yoon*, Uehwan Kim* and Jong-Hwan Kim},
  journal={IEEE Transactions on Neural Networks and Learning Systems, Under Review},
  year={2019}
}
```

## 2. Stabilized-Feedback Episodic Memory (SF-EM)
Episodic memory incrementally learns user behaviors and event sequences.
However, conventional episodic memory fails to stably perform over a long period of time.
In addition, they cannot not accept user feedback.
The proposed SF-EM stably performs over a long period of time and accepts user feedback.
The following are the key features of SF-EM:

- An adaptive decay factor to enhance the stability of the learning process of the memory architecture.
- A feedback mechanism to reflect user feedback.
- A home service provision framework for robot and IoT collaboration.

For detailed formulation of SF-EM, please refer to [the SFEM directory](SFEM) of our repository and the following paper:
```
@article{kim2018a,
  title={A Stabilized Feedback Episodic Memory (SF-EM) and Home Service Provision Framework for Robot and IoT Collaboration},
  author={Uehwan Kim and Jong-Hwan Kim},
  journal={IEEE Transactions on Cybernetics, Early Access},
  year={2018}
}
```

## References
This repository contains implementation of following works as components for the proposed s-DRN and SF-EM:
- G. A. Carpenter, S. Grossberg, and D. B. Rosen, “Fuzzy ART: An adaptive resonance algorithm for rapid, stable classification of analog patterns,” in Proc. Int. Joint Conf. Neural Netw., vol. 2, 1991, pp. 411–416.
- W. Wang, B. Subagdja, A.-H. Tan, and J. A. Starzyk, “Neural modeling of episodic memory: Encoding, retrieval, and forgetting,” IEEE Trans. Neural Netw. Learn. Syst., vol. 23, no. 10, pp. 1574–1586, Oct. 2012.
- G.-M. Park, Y.-H. Yoo, D.-H. Kim, and J.-H. Kim, “Deep ART neural model for biologically inspired episodic memory and its application to task performance of robots,” IEEE Trans. Cybern., vol. 48, no. 6, pp. 1786–1799, Jun. 2018.

## Acknowledgments
This work was supported by Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT)
(No.2016-0-00563, Research on Adaptive Machine Learning Technology Development for Intelligent Autonomous Digital Companion)