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
  -  Mapping
    <img src="./png/ICA_components.png" alt="Alt text" title="Battle ship" style="display: inline-block; max-width: 20px">
  - ica.fit() in MNE
    - whitening
      -     Covariance Matrix Estimation: the covariance matrix of the observed data is estimated.
            This matrix describes the statistical relationships (covariances) between different channels
            or features.
      -     Whitening Transformation: The estimated covariance matrix is used to perform a transformation
            on the data that "whitens" it. Whitening is a linear transformation that makes the transformed
            data have an identity covariance matrix, i.e., the transformed data has uncorrelated components
            and unit variance. This is often achieved using methods like Cholesky decomposition or the
            square root of the inverse of the covariance matrix.

    - Passing the n_components largest-variance components to the ICA algorithm to obtain the unmixing matrix 

  #Dimension Reduction  
    - Correlation
    -  χ² (Khi²)
  #Pipeline object scikit-learn

  #Data stream classification in real time
