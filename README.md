# AU8: A Multimodal Benchmark Dataset from Eight Australian Cities for Sustainable Urban Development

This repository contains the source code and scripts for the AU8 benchmark dataset.

[Dataset on Hugging Face](https://huggingface.co/datasets/CRUISEResearchGroup/AU8)

📦 Dataset Overview
AU8 is a large-scale multimodal benchmark dataset designed to support research on urban indicator prediction and sustainable urban development. It provides spatially aligned satellite imagery, text descriptions, and structured urban indicators for eight major Australian cities.

🗺️ Key Features
📍 8 Cities: Greater Sydney, Melbourne, Brisbane, Perth, Adelaide, Canberra, Darwin, Hobart

🛰️ 101,604 Satellite Images: Downloaded from Google Earth (not publicly released due to license)

📄 Text Descriptions: Automatically generated using GPT-4o to describe terrain, vegetation, water bodies, and human-made structures

📊 9 Urban Indicators:

Population & Employment:

Population_density

Number_of_jobs

Number_of_employed_persons

Socioeconomic:

Median_total_income

Number_of_businesses

Median_house_price

Land Use:

Protected_land_area

Agricultural_land_area

Rural_farm_infrastructure_area

📐 Data Format
Each sample in the dataset includes:

Component	Type	Description
image_name	Image	Filename of the satellite image (e.g., satellite_image_-33.8_151.2.png)
latitude, longitude	Metadata	WGS84 center coordinate of the image tile
city	Metadata	City name (e.g., Sydney, Melbourne)
SA2_CODE_2021	Metadata	ABS-defined Statistical Area 2 code
caption	Text	GPT-4o-generated declarative description
indicator_1 ~ 9	Numeric	Urban indicators (log-transformed for regression modeling)
