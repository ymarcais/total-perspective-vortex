  # Total-Perspective-Vortex
  This subject aims to create a brain computer interface based on electroencephalographic data (EEG data) with the help of machine learning algorithms. Using a subject’s EEG reading, it infers what he or she is thinking about or doing - (motion) A or B in a t0 to tn timeframe. The data was mesured during a motor imagery experiment, where people had to do or imagine a hand or feet movement.
  
  #Environment - conda activate tpv - 
  - numpy
  - pandas
  - mne!=0
  
  #Data
    - experimental protocol
        https://physionet.org/content/eegmmidb/1.0.0/
    - loading data  
    <img src="./png/raw_sample_S001.png" alt="Alt text" title="Battle ship" style="display: inline-block; max-width: 20px">
    

  #Parsing and filtering
  Mapping
    <img src="./png/ICA_components.png" alt="Alt text" title="Battle ship" style="display: inline-block; max-width: 20px">
  
  #Dimension Reduction  
    - Correlation
    -  χ² (Khi²)
  #Pipeline object scikit-learn

  #Data stream classification in real time
