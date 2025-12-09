# Modeling Drosophila Sleep Activity Data using VAEs

Goal: Model drosophila ethoscope data using a VAE.  

Model Architecture: Conditional β-VAE  

Data: 
- Raw sleep data from a conditional expression experiment where the addition of a drug will induce expression of the AD model
  - Condition: “RU” or “RU486” - AD fly (1)
  - Condition: “EtOH” - Control fly (0)
- 150 samples, beam breaks measured every minute. Truncated to 20068 timepoints (1 min bins).

