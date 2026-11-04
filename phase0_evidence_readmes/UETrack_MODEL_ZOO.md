# Model Zoo

Here we provide the performance of UETrack on multiple tracking benchmarks and the corresponding raw results.  
The model weights, logs, and raw tracking outputs can be downloaded from the links below.

UETrack is a unified and efficient framework for single object tracking that supports five modalities: **RGB, Depth, Thermal, Event, and Language**. It includes three model variants: **UETrack-B**, **UETrack-S**, and **UETrack-T**.

## UETrack Models

<table>
  <tr>
    <th>Model</th>
    <th>LaSOT<br>AUC (%)</th>
    <th>LaSOText<br>AUC (%)</th>
    <th>TrackingNet<br>AUC (%)</th>
    <th>GOT-10k<br>AO (%)</th>
    <th>VOT-RGBD22<br>EAO (%)</th>
    <th>DepthTrack<br>F-score (%)</th>
    <th>LasHeR<br>AUC (%)</th>
    <th>RGBT234<br>MSR (%)</th>
    <th>VisEvent<br>AUC (%)</th>
    <th>TNL2K<br>AUC (%)</th>
    <th>OTB99<br>AUC (%)</th>
    <th>GPU/CPU/AGX<br>FPS</th>
    <th>Params<br>(M)</th>
    <th>FLOPs<br>(G)</th>
    <th>Link</th>
  </tr>
  <tr>
    <td>UETrack-B</td>
    <td>69.2</td>
    <td>48.4</td>
    <td>82.7</td>
    <td>72.6</td>
    <td>68.3</td>
    <td>60.6</td>
    <td>55.5</td>
    <td>64.2</td>
    <td>59.2</td>
    <td>58.0</td>
    <td>61.3</td>
    <td>163/56/60</td>
    <td>13</td>
    <td>3.2</td>
    <td><a href="https://huggingface.co/kangben258/UETrack">[Download]</a></td>
  </tr>
  <tr>
    <td>UETrack-S</td>
    <td>66.9</td>
    <td>47.9</td>
    <td>81.4</td>
    <td>71.1</td>
    <td>66.5</td>
    <td>58.9</td>
    <td>53.2</td>
    <td>62.2</td>
    <td>58.0</td>
    <td>57.0</td>
    <td>63.1</td>
    <td>183/68/67</td>
    <td>9</td>
    <td>2.5</td>
    <td><a href="https://huggingface.co/kangben258/UETrack">[Download]</a></td>
  </tr>
  <tr>
    <td>UETrack-T</td>
    <td>63.4</td>
    <td>42.2</td>
    <td>78.9</td>
    <td>65.3</td>
    <td>62.5</td>
    <td>55.7</td>
    <td>48.2</td>
    <td>59.3</td>
    <td>54.4</td>
    <td>54.4</td>
    <td>64.8</td>
    <td>221/83/77</td>
    <td>6</td>
    <td>1.8</td>
    <td><a href="https://huggingface.co/kangben258/UETrack">[Download]</a></td>
  </tr>
</table>

