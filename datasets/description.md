# Dataset Description
* 1.ETT (Electricity Transformer Temperature): The ETT is a crucial indicator in the electric power long-term deployment. We collected 2-year data from two separated counties in China. To explore the granularity on the LSTF problem,we create separate datasets as {ETTh1, ETTh2} for 1-hour level and ETTm1 for 15-minute-level. Each data point consists of the target value ”oil temperature” and 6 power load features. The train/val/test is 12/4/4 months.

* 2.Electricity: The raw dataset is in https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014. It is the electricity consumption in kWh was recorded every 15 minutes from 2011 to 2014. Because the some dimensions are equal to 0. So we eliminate the records in 2011. Final we get data contains electircity consumption of 321 clients from 2012 to 2014. And we converted the data to reflect hourly consumption.

* 3.Exchange Rate: the collection of the daily exchange rates of eight foreign countries including Australia, British, Canada, Switzerland, China, Japan, New Zealand and Singapore ranging from 1990 to 2016.

* 4.Traffic: The raw data is in http://pems.dot.ca.gov. The data in this repo is a collection of 48 months (2015-2016) hourly data from the California Department of Transportation. The data describes the road occupancy rates (between 0 and 1) measured by different sensors on San Francisco Bay area freeways.

* 5.Weather: is recorded every 10 minutes for 2020 whole year, which contains 21 meteorological indicators, such as air temperature, humidity, etc.

* 6.ILI: includes the weekly recorded influenza-like illness (ILI) patients data from Centers for Disease Control and Prevention of the United States between 2002 and 2021, which describes the ratio of patients seen with ILI and the total number of the patients.
