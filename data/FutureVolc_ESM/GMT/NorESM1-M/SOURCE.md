**Dataset:** NorESM1-M RCP4.5 large ensemble, with possible future volcanic eruptions: GMST with stochastic future eruptions (dataset ID FutureVolc_ESM-rcp45-Volc) and with constant background volcanic forcing (FutureVolc_ESM-rcp45-VolcConst)

**Cite:**
- Data: Thorne, P., & Nicklas, J. (2026). Trimmed model and observational inputs for "How will we know when 1.5°C of global warming has been reached?" (Version v1.0-esd-submission) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.23568301
- Ensemble: Bethke, I., et al. (2017). Potential volcanic impacts on future climate variability. Nature Climate Change, 7, 799–805. https://doi.org/10.1038/nclimate3394

**Licence:** Creative Commons Attribution 4.0 International

**From:** Supplied by John Nicklas

**Notes:** Model output, so the ERF and GMST pair only with each other, for the same variant: never with observations or another model.
- FutureVolc_ESM-rcp45-Volc: each of the 60 GMST members has its own natural (volcanic) ERF member, with the same label. Attribute them one-to-one, one member at a time (ERF member 1 with GMST member 1, and so on).
- FutureVolc_ESM-rcp45-VolcConst: the ERF is a single timeseries shared by all 60 GMST members.

**History and projection:** CMIP5 historical simulations to 2005, then RCP4.5 from 2006. The GMST of each member is identical between rcp45-Volc and rcp45-VolcConst up to 2005, and the anthropogenic ERF is identical throughout. The natural ERF is not identical before 2006: the 60 rcp45-Volc members differ from each other, and from rcp45-VolcConst, by up to about 0.06 W/m² (e.g. in 1990), against 1–5 W/m² after 2006. Still to check whether this is expected.
