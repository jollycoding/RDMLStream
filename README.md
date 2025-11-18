# RDMLStream
Region-based Deep Metric Learning for Tackling Class Overlap in Online Semi-Supervised Data Stream Classification

**Abstract**  
Class overlap in data streams presents a significant challenge for real-time classification, particularly when confronted with the high dimensionality and evolving distributions inherent in such streams. Traditional classification methods, typically designed for static datasets, struggle to adapt to the dynamic nature of data streams, where both high-dimensional feature spaces and class imbalance exacerbate the complexity of classifying overlapping regions. In this paper, we propose a novel deep metric learning framework specifically tailored to address the challenges of class overlap in high-dimensional data streams. Our approach introduces two key innovations. First, we develop a multi-anchor sample mining mechanism based on neighborhood rough set theory, which partitions the data into non-overlapping and overlapping regions. By utilizing region-specific triplet-margin losses and hinge embedding loss, we construct a more refined discriminative metric space that significantly enhances the separation of overlapping classes. Furthermore, we introduce a dynamic, density-aware real-time label propagation mechanism with class-imbalance compensation. This component integrates real-time distribution estimation with a nonlinear adaptive threshold controller, enabling dual adaptivity: (1) dynamically re-weighting density contributions via inverse-frequency scaling to mitigate the dominance of majority classes and (2) adjusting threshold boundaries for frequent classes while relaxing propagation criteria for rare classes through nonlinear adjustments. Empirical evaluations on both synthetic and real-world data streams demonstrate that our method not only improves balanced accuracy but also enhances robustness in the presence of class overlap and class imbalance, outperforming state-of-the-art techniques.

**Synthetic Dataset Generation**  

The synthetic datasets used in this project (`Wave_Su` and `Wave_Gr`) were generated using **MOA (Massive Online Analysis)**. MOA is an open-source framework for data stream mining and is particularly well-suited for handling concept drift and large-scale datasets.

You can replicate or regenerate these datasets by following the steps below.

### Prerequisites

1.  **Java Environment**: Ensure you have a Java (JDK) environment installed.
2.  **MOA**: Download `moa.jar` from the [official MOA website](https://moa.cms.waikato.ac.nz/downloads/).
3.  **Sizeof Agent**: Download the `sizeofag.jar` (e.g., `sizeofag-1.0.4.jar`), which is typically provided alongside the MOA download.

Place both `moa.jar` and `sizeofag-1.0.4.jar` in your working directory.

### Dataset Generation

We use the `WriteStreamToARFFFile` task in MOA, combined with `ConceptDriftStream`, `ImbalancedStream`, and `WaveformGeneratorDrift`. This setup allows us to generate data streams with specific class imbalance ratios and concept drift, saving the output to ARFF files.

-----

#### 1\. Wave\_Su (Sudden Drift)

This dataset simulates a **sudden concept drift** under **class imbalance**.

  * **Total Instances**: 50,000
  * **Base Generator**: `generators.WaveformGeneratorDrift`
  * **Drift Type**: Sudden
  * **Drift Point**: Occurs at the 5,000th instance (`-p 5000`)
  * **Drift Window**: 1 (`-w 1`), indicating an instantaneous change.
  * **Imbalance Ratio Change**:
      * **Before Drift**: Class ratios are `0.02:0.2:0.78` (approx. 1:10:40)
      * **After Drift**: Class ratios change to `0.78:0.02:0.2` (approx. 40:1:10)

**Generation Command:**

```bash
java -cp moa.jar -javaagent:sizeofag-1.0.4.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (ImbalancedStream -s (generators.WaveformGeneratorDrift -d 10 -n)-c 0.02;0.2;0.78) -d (ImbalancedStream -s (generators.WaveformGeneratorDrift -d 10 -n)-c 0.78;0.02;0.2) -p 5000 -w 1) -f ./Wave_sudden.arff -m 50000"
```


#### 2\. Wave\_Gr (Gradual Drift)

This dataset simulates a **gradual concept drift** under **class imbalance**.

  * **Total Instances**: 50,000
  * **Base Generator**: `generators.WaveformGeneratorDrift`
  * **Drift Type**: Gradual
  * **Drift Point**: Begins at the 10,000th instance (`-p 10000`)
  * **Drift Window**: 5,000 (`-w 5000`), meaning the concept transition completes over a period of 5,000 instances.
  * **Imbalance Ratio Change**:
      * **Before Drift**: Class ratios are `0.02:0.2:0.78` (approx. 1:10:40)
      * **After Drift**: Class ratios change to `0.78:0.02:0.2` (approx. 40:1:10)

**Generation Command:**

```bash
java -cp moa.jar -javaagent:sizeofag-1.0.4.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (ImbalancedStream -s (generators.WaveformGeneratorDrift -d 10 -n)-c 0.02;0.2;0.78) -d (ImbalancedStream -s (generators.WaveformGeneratorDrift -d 10 -n)-c 0.78;0.02;0.2) -p 10000 -w 5000) -f ./Wave_gradual.arff -m 50000"
```
