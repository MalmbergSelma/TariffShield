Description of the files for the paper "The Macroeconomic and Redistributive Eﬀects of the French Tariﬀ Shield"
by François Langot, Selma Malmberg, Fabien Tripier and Jean-Olivier Hairault

CepreHANKv1.py
	File defining the equations of the model to ensure consistency across the other files
	Contains the household block, the steady state equations and dynamic ones

TariffShield_Estimation.ipynb
	Code to do historical estimation of the shocks' parameters + account for the Lucas critique + generate the scenarios (tariff shield, no tariff shield, wage indexation, transfers)
	Elements to replicate: - Tables 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22
					       - Figures 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 18

TariffShield_IRFs.ipynb
	Code to generate linear and non-linear IRFs to an energy price shock or a tariff shield shock
	Elements to replicate: - Tables 2, 17
					       - Figures 1, 2, 16, 17, 19

TariffShield_AltModels.ipynb
	Code for alternative modelisation choices for the energy market and trade balance
	Elements to replicate: - Tables 10, 11, 23, 24, 25
					       - Figures 20, 21, 22, 23

esti_data.xlsx
	Data used for the historical estimation

data_PLF2023.xlsx
	Government forecasts for the 2023 Finance Act

The figures and estimation results are saved into the Figures and Results folders.