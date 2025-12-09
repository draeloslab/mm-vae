# Modeling Drosophila Sleep Activity Data using VAEs

Goal: Model drosophila ethoscope data using a VAE.
Data: 
- Raw sleep data from a conditional expression experiment where the addition of a drug will induce expression of the AD model
- Condition: “RU” or “RU486” - AD fly
- Condition: “EtOH” - Control fly
- 150 samples, beam breaks measured every minute. Truncated to 20068 timepoints (1 min bins).
Model Architecture: Conditional β-VAE
