# LADM: Long-context Training Data Selection with Attention-based Dependency Measurement for LLMs
<p align="center">
    📖 <a href="https://arxiv.org/abs/2503.02502" target="_blank">Paper</a> • 🤗 <a href="https://huggingface.co/collections/UltraRonin/ladm-68466cbccb652c8d828ca17e" target="_blank">HF Repo</a>
</p>

## 🔍 Table of Contents
- [🌐 Overview](#overview)
- [🤖️ AgentWrite](#agentwrite)
- [🖥️ Model Training](#longwriter-training)
- [📊 Evaluation](#evaluation)
- [👀 Cases](#case)
- [📝 Citation](#citation)


<a name="overview"></a>
Long-context modeling has drawn more and more attention in the area of Large Language Models (LLMs). Continual training with long-context data becomes the de-facto method to equip LLMs with the ability to process long inputs. However, it still remains an open challenge to measure the quality of long-context training data. To address this issue, we propose a Long-context data selection framework with Attention-based Dependency Measurement (LADM), which can efficiently identify high-quality long-context data from a large-scale, multi-domain pre-training corpus. LADM leverages the retrieval capabilities of the attention mechanism to capture contextual dependencies, ensuring a comprehensive quality measurement of long-context data. Experimental results show that our LADM framework significantly boosts the performance of LLMs on multiple long-context tasks with only 1B tokens for continual training.
![](./assets/framework.png)
