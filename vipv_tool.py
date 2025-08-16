import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import requests
import datetime
import time
from bs4 import BeautifulSoup

plt.style.use('seaborn-v0_8-darkgrid')
COLOR_PALETTE = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B"]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=COLOR_PALETTE)

# Data setup
cities = ['Barcelona', 'Berlin', 'Cairo', 'Delhi', 'Dubai', 'London', 'Madrid', 'Melbourne',
          'Milan', 'Mumbai', 'Paris', 'Riyadh', 'Rome', 'Seville', 'Sydney']

irradiation_data = {
    'Barcelona': [2.874, 3.471, 4.099, 4.810, 4.693, 4.040, 3.925, 4.428, 4.074, 4.303, 3.671, 3.514],
    'Berlin': [1.073, 1.826, 2.670, 3.808, 4.213, 4.194, 3.986, 3.641, 2.953, 1.989, 1.141, 0.895],
    'Cairo': [3.914, 4.258, 5.054, 5.361, 6.451, 7.358, 7.315, 6.912, 5.996, 4.966, 4.310, 4.056],
    'Delhi': [2.576, 3.838, 4.603, 4.585, 3.548, 2.612, 1.772, 2.253, 3.329, 3.542, 2.941, 2.777],
    'Dubai': [4.834, 4.974, 5.070, 5.252, 5.949, 5.514, 4.190, 4.635, 5.433, 5.802, 5.344, 4.864],
    'London': [1.176, 1.737, 2.352, 3.256, 3.526, 3.593, 3.573, 3.064, 2.607, 1.954, 1.428, 1.064],
    'Madrid': [3.704, 4.525, 5.242, 5.511, 6.142, 7.491, 8.500, 7.425, 5.857, 4.480, 3.655, 3.402],
    'Melbourne': [6.580, 5.668, 4.857, 3.693, 2.986, 2.773, 2.972, 3.319, 4.031, 4.623, 5.091, 6.353],
    'Milan': [2.635, 3.396, 4.420, 4.460, 4.728, 5.330, 5.985, 5.173, 4.086, 2.714, 2.152, 2.237],
    'Mumbai': [5.214, 5.825, 5.751, 5.452, 5.007, 2.252, 0.998, 1.334, 2.354, 3.715, 4.569, 4.731],
    'Paris': [1.328, 2.019, 2.920, 3.821, 3.976, 4.282, 4.232, 3.896, 3.403, 2.234, 1.529, 1.248],
    'Riyadh': [5.141, 5.311, 5.202, 4.573, 5.170, 6.239, 6.083, 6.031, 6.529, 6.915, 5.467, 5.411],
    'Rome': [2.962, 3.733, 4.265, 4.617, 5.579, 6.603, 7.287, 6.349, 4.658, 3.653, 2.765, 2.786],
    'Seville': [4.053, 4.632, 5.309, 5.756, 6.520, 7.627, 8.230, 7.352, 5.874, 4.639, 4.176, 3.834],
    'Sydney': [5.378, 4.621, 4.608, 4.568, 4.747, 3.643, 4.762, 5.327, 5.719, 5.373, 5.053, 5.229]
}

energy_cost = {
    'Barcelona': 0.25, 'Berlin': 0.35, 'Cairo': 0.10, 'Delhi': 0.08, 'Dubai': 0.12,
    'London': 0.30, 'Madrid': 0.22, 'Melbourne': 0.20, 'Milan': 0.28, 'Mumbai': 0.09,
    'Paris': 0.32, 'Riyadh': 0.11, 'Rome': 0.26, 'Seville': 0.21, 'Sydney': 0.19
}

# City coordinates for Solarcast API
city_coordinates = {
    'Barcelona': {'lat': 41.3851, 'lon': 2.1734},
    'Berlin': {'lat': 52.5200, 'lon': 13.4050},
    'Cairo': {'lat': 30.0444, 'lon': 31.2357},
    'Delhi': {'lat': 28.7041, 'lon': 77.1025},
    'Dubai': {'lat': 25.2048, 'lon': 55.2708},
    'London': {'lat': 51.5074, 'lon': -0.1278},
    'Madrid': {'lat': 40.4168, 'lon': -3.7038},
    'Melbourne': {'lat': -37.8136, 'lon': 144.9631},
    'Milan': {'lat': 45.4642, 'lon': 9.1900},
    'Mumbai': {'lat': 19.0760, 'lon': 72.8777},
    'Paris': {'lat': 48.8566, 'lon': 2.3522},
    'Riyadh': {'lat': 24.7136, 'lon': 46.6753},
    'Rome': {'lat': 41.9028, 'lon': 12.4964},
    'Seville': {'lat': 37.3891, 'lon': -5.9845},
    'Sydney': {'lat': -33.8688, 'lon': 151.2093}
}

# Tutiempo.net URLs for Spanish cities
tutiempo_urls = {
    'Barcelona': 'https://www.tutiempo.net/radiacion-solar/barcelona.html',
    'Madrid': 'https://www.tutiempo.net/radiacion-solar/madrid.html',
    'Seville': 'https://www.tutiempo.net/radiacion-solar/sevilla.html'
}

# Realistic angle data for each surface type (in degrees)
surface_angles = {
    'hood': 15,    # Hoods are typically slightly angled
    'roof': 5,     # Roofs are nearly flat
    'rear_window': 45,  # Rear windows are steeply angled
    'rear_side_window': 30,  # Rear side windows
    'front_side_window': 25,  # Front side windows
    'canopy': 0     # Canopies are typically flat
}

# Complete segments dictionary
segments = {
    'B-HB (Micra)': {
        'wltp': 12.5,
        'city': 9.4,
        'surfaces': {
            'hood': {'area': 1.2, 'angle': surface_angles['hood']},
            'roof': {'area': 1.5, 'angle': surface_angles['roof']},
            'rear_window': {'area': 0.4, 'angle': surface_angles['rear_window']},
            'rear_side_window': {'area': 0.6, 'angle': surface_angles['rear_side_window']},
            'front_side_window': {'area': 0.5, 'angle': surface_angles['front_side_window']},
            'canopy': {'area': 0, 'angle': surface_angles['canopy'], 'default': False}
        }
    },
    'B-SUV (Juke)': {
        'wltp': 13.5,
        'city': 10.1,
        'surfaces': {
            'hood': {'area': 1.4, 'angle': surface_angles['hood']},
            'roof': {'area': 1.8, 'angle': surface_angles['roof']},
            'rear_window': {'area': 0.5, 'angle': surface_angles['rear_window']},
            'rear_side_window': {'area': 0.7, 'angle': surface_angles['rear_side_window']},
            'front_side_window': {'area': 0.6, 'angle': surface_angles['front_side_window']},
            'canopy': {'area': 0, 'angle': surface_angles['canopy'], 'default': False}
        }
    },
    'C-HB (Leaf)': {
        'wltp': 13.0,
        'city': 9.8,
        'surfaces': {
            'hood': {'area': 1.5, 'angle': surface_angles['hood']},
            'roof': {'area': 2.0, 'angle': surface_angles['roof']},
            'rear_window': {'area': 0.6, 'angle': surface_angles['rear_window']},
            'rear_side_window': {'area': 0.8, 'angle': surface_angles['rear_side_window']},
            'front_side_window': {'area': 0.7, 'angle': surface_angles['front_side_window']},
            'canopy': {'area': 0, 'angle': surface_angles['canopy'], 'default': False}
        }
    },
    'C-SUV (Qashqai)': {
        'wltp': 14.5,
        'city': 10.9,
        'surfaces': {
            'hood': {'area': 1.6, 'angle': surface_angles['hood']},
            'roof': {'area': 2.2, 'angle': surface_angles['roof']},
            'rear_window': {'area': 0.7, 'angle': surface_angles['rear_window']},
            'rearside_window': {'area': 0.9, 'angle': surface_angles['rear_side_window']},
            'front_side_window': {'area': 0.8, 'angle': surface_angles['front_side_window']},
            'canopy': {'area': 0, 'angle': surface_angles['canopy'], 'default': False}
        }
    },
    'C-SUV+ (X-Trail)': {
        'wltp': 15.0,
        'city': 11.3,
        'surfaces': {
            'hood': {'area': 1.7, 'angle': surface_angles['hood']},
            'roof': {'area': 2.4, 'angle': surface_angles['roof']},
            'rear_window': {'area': 0.8, 'angle': surface_angles['rear_window']},
            'rear_side_window': {'area': 1.0, 'angle': surface_angles['rear_side_window']},
            'front_side_window': {'area': 0.9, 'angle': surface_angles['front_side_window']},
            'canopy': {'area': 0, 'angle': surface_angles['canopy'], 'default': False}
        }
    },
    'D-SUV (X-Terra)': {
        'wltp': 16.0,
        'city': 12.0,
        'surfaces': {
            'hood': {'area': 1.8, 'angle': surface_angles['hood']},
            'roof': {'area': 2.6, 'angle': surface_angles['roof']},
            'rear_window': {'area': 0.9, 'angle': surface_angles['rear_window']},
            'rear_side_window': {'area': 1.1, 'angle': surface_angles['rear_side_window']},
            'front_side_window': {'area': 1.0, 'angle': surface_angles['front_side_window']},
            'canopy': {'area': 0, 'angle': surface_angles['canopy'], 'default': False}
        }
    },
    'D-SDN (Altima)': {
        'wltp': 13.5,
        'city': 10.1,
        'surfaces': {
            'hood': {'area': 1.7, 'angle': surface_angles['hood']},
            'roof': {'area': 2.3, 'angle': surface_angles['roof']},
            'rear_window': {'area': 0.7, 'angle': surface_angles['rear_window']},
            'rear_side_window': {'area': 0.9, 'angle': surface_angles['rear_side_window']},
            'front_side_window': {'area': 0.8, 'angle': surface_angles['front_side_window']},
            'canopy': {'area': 0, 'angle': surface_angles['canopy'], 'default': False}
        }
    },
    'E-SUV (Pathfinder)': {
        'wltp': 17.0,
        'city': 12.8,
        'surfaces': {
            'hood': {'area': 2.0, 'angle': surface_angles['hood']},
            'roof': {'area': 2.8, 'angle': surface_angles['roof']},
            'rear_window': {'area': 1.0, 'angle': surface_angles['rear_window']},
            'rear_side_window': {'area': 1.2, 'angle': surface_angles['rear_side_window']},
            'front_side_window': {'area': 1.1, 'angle': surface_angles['front_side_window']},
            'canopy': {'area': 0, 'angle': surface_angles['canopy'], 'default': False}
        }
    },
    'F-SUV (Patrol)': {
        'wltp': 18.0,
        'city': 13.5,
        'surfaces': {
            'hood': {'area': 2.2, 'angle': surface_angles['hood']},
            'roof': {'area': 3.0, 'angle': surface_angles['roof']},
            'rear_window': {'area': 1.1, 'angle': surface_angles['rear_window']},
            'rear_side_window': {'area': 1.3, 'angle': surface_angles['rear_side_window']},
            'front_side_window': {'area': 1.2, 'angle': surface_angles['front_side_window']},
            'canopy': {'area': 0, 'angle': surface_angles['canopy'], 'default': False}
        }
    },
    'Mid-VAN (NV200)': {
        'wltp': 18.0,
        'city': 13.5,
        'surfaces': {
            'hood': {'area': 1.8, 'angle': surface_angles['hood']},
            'roof': {'area': 3.2, 'angle': surface_angles['roof']},
            'rear_window': {'area': 1.0, 'angle': surface_angles['rear_window']},
            'rear_side_window': {'area': 1.4, 'angle': surface_angles['rear_side_window']},
            'front_side_window': {'area': 1.0, 'angle': surface_angles['front_side_window']},
            'canopy': {'area': 0, 'angle': surface_angles['canopy'], 'default': False}
        }
    },
    'Pick Up (Navara)': {
        'wltp': 20.0,
        'city': 15.0,
        'surfaces': {
            'hood': {'area': 2.0, 'angle': surface_angles['hood']},
            'roof': {'area': 2.5, 'angle': surface_angles['roof']},
            'rear_window': {'area': 0.8, 'angle': surface_angles['rear_window']},
            'rear_side_window': {'area': 1.0, 'angle': surface_angles['rear_side_window']},
            'front_side_window': {'area': 0.9, 'angle': surface_angles['front_side_window']},
            'canopy': {'area': 4.0, 'angle': surface_angles['canopy'], 'default': True}
        }
    }
}

# Default values
default_utilization = 90
default_pv_efficiency = 25
default_cost = 350
default_transformation_efficiency = 90

# Solarcast API functions with robust error handling
def get_solarcast_forecast(api_key, latitude, longitude):
    """Fetch solar forecast data from Solarcast API with robust error handling"""
    base_url = "https://api.solarcast.io/forecast"
    params = {
        "lat": latitude,
        "lon": longitude,
        "apikey": api_key
    }
    
    try:
        # Increased timeout to 15 seconds for read operations
        response = requests.get(base_url, params=params, timeout=(5, 15))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        st.warning("Solarcast API timed out. This might be due to network issues or high API load. Using monthly averages instead.")
        return None
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not retrieve solar forecast: {str(e)}. Using monthly averages.")
        return None
    except Exception as e:
        st.warning(f"Unexpected error: {str(e)}. Using monthly averages.")
        return None

def extract_forecast_days(forecast_data):
    """Extract forecast for the next 5 days from API response"""
    if not forecast_data or "daily" not in forecast_data:
        return None
    
    today = datetime.date.today()
    forecast_days = {}
    
    for i in range(6):  # Today + next 5 days
        date_str = (today + datetime.timedelta(days=i)).isoformat()
        if date_str in forecast_data["daily"]:
            day_label = "Today" if i == 0 else f"Day +{i}"
            forecast_days[day_label] = forecast_data["daily"][date_str]["solar_irradiance"]
    
    return forecast_days

# Tutiempo.net web scraping for Spanish cities
def get_tutiempo_forecast(city):
    """Get solar irradiation forecast from Tutiempo.net for Spanish cities"""
    try:
        url = tutiempo_urls.get(city)
        if not url:
            return None
            
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the table with forecast data
        table = soup.find('table', class_='medias')
        if not table:
            return None
            
        # Extract the next 15 days of solar irradiation
        forecast = {}
        rows = table.find_all('tr')[1:16]  # Next 15 days
        
        today = datetime.date.today()
        for i, row in enumerate(rows):
            cols = row.find_all('td')
            if len(cols) >= 5:
                # Extract solar radiation value
                radiation = cols[4].get_text().strip()
                try:
                    radiation_value = float(radiation)
                except ValueError:
                    radiation_value = None
                    
                # Create day label
                day_label = "Today" if i == 0 else f"Day +{i}"
                forecast_date = today + datetime.timedelta(days=i)
                
                forecast[day_label] = {
                    'date': forecast_date.strftime("%Y-%m-%d"),
                    'radiation': radiation_value
                }
                
        return forecast
        
    except Exception as e:
        st.warning(f"Could not retrieve Tutiempo forecast: {str(e)}")
        return None

# Streamlit app
st.set_page_config(layout="wide", page_title="VIPV Evaluation Tool")
st.title("VIPV Evaluation Tool")

# Create tabs
tab1, tab2 = st.tabs(["Assumptions", "Visualization"])

with tab1:
    st.header("Assumptions")

    col1, col2 = st.columns(2)

    with col1:
        # Region selection
        region = st.selectbox("Select Region", cities, index=cities.index('Dubai'))
        
        # Irradiation source selection
        irradiation_options = ["Monthly Average", "Solarcast API Forecast"]
        
        # Add Tutiempo option only for Spanish cities
        if region in tutiempo_urls:
            irradiation_options.append("Tutiempo 15-Day Forecast")
            
        irradiation_source = st.radio("Irradiation Data Source", 
                                    irradiation_options, 
                                    index=0,
                                    help="Select between historical monthly averages or real-time solar forecasts")
        
        # Initialize variables
        avg_irradiation = np.mean(irradiation_data[region])
        daily_irradiation = avg_irradiation
        forecast_data = None
        forecast_days = None
        selected_day = "Monthly Average"
        data_source = "Monthly Average"
        
        # Solarcast API integration
        if irradiation_source == "Solarcast API Forecast":
            api_key = st.text_input("Solarcast API Key", type="password",
                                   help="Get your API key from solarcast.io")
            
            # Get coordinates for selected region
            coords = city_coordinates.get(region)
            if not coords:
                st.warning(f"Coordinates not available for {region}. Using Dubai coordinates.")
                coords = city_coordinates['Dubai']
            
            # Fetch forecast data
            if api_key:
                with st.spinner("Fetching solar forecast..."):
                    # Try API call with retry
                    for attempt in range(3):  # Try up to 3 times
                        forecast_data = get_solarcast_forecast(api_key, coords['lat'], coords['lon'])
                        if forecast_data is not None:
                            break
                        if attempt < 2:  # Not the last attempt
                            time.sleep(1)  # Wait before retrying
                
                if forecast_data:
                    forecast_days = extract_forecast_days(forecast_data)
                    if forecast_days:
                        selected_day = st.selectbox("Select Forecast Day", list(forecast_days.keys()))
                        daily_irradiation = forecast_days[selected_day]
                        st.success(f"Using {selected_day} forecast: {daily_irradiation:.2f} kWh/m²/day")
                        data_source = f"Solarcast API ({selected_day})"
                    else:
                        st.warning("Forecast data format not recognized. Using monthly average.")
                        st.metric("Average Daily Irradiation", f"{avg_irradiation:.2f} kWh/m²/day")
                else:
                    st.metric("Average Daily Irradiation", f"{avg_irradiation:.2f} kWh/m²/day")
            else:
                st.info("Please enter your Solarcast API key to get real-time forecasts")
                st.metric("Average Daily Irradiation", f"{avg_irradiation:.2f} kWh/m²/day")
        
        # Tutiempo integration for Spanish cities
        elif irradiation_source == "Tutiempo 15-Day Forecast":
            with st.spinner(f"Fetching solar forecast for {region} from Tutiempo.net..."):
                forecast_data = get_tutiempo_forecast(region)
                
            if forecast_data:
                # Create list of available days with radiation values
                available_days = [day for day, data in forecast_data.items() if data['radiation'] is not None]
                
                if available_days:
                    selected_day = st.selectbox("Select Forecast Day", available_days)
                    daily_irradiation = forecast_data[selected_day]['radiation']
                    forecast_date = forecast_data[selected_day]['date']
                    st.success(f"Using forecast for {forecast_date}: {daily_irradiation:.2f} kWh/m²/day")
                    data_source = f"Tutiempo ({forecast_date})"
                else:
                    st.warning("No radiation data available in forecast. Using monthly average.")
                    st.metric("Average Daily Irradiation", f"{avg_irradiation:.2f} kWh/m²/day")
            else:
                st.metric("Average Daily Irradiation", f"{avg_irradiation:.2f} kWh/m²/day")
        else:
            st.metric("Average Daily Irradiation", f"{avg_irradiation:.2f} kWh/m²/day")
            
        st.divider()
        st.metric("Default Electricity Price", f"{energy_cost[region]:.2f} €/kWh")
        electricity_price = st.slider("Adjust Electricity Price (€/kWh)",
                            min_value=0.01, max_value=1.0,
                            value=energy_cost[region], step=0.01)
    with col2:
        # Segment selection
        segment = st.selectbox("Select Segment", list(segments.keys()))
        segment_data = segments[segment]
        col2a, col2b = st.columns(2)
        with col2a:
            st.metric("WLTP Efficiency", f"{segment_data['wltp']} kWh/100km")
        with col2b:
            st.metric("City Efficiency", f"{segment_data['city']} kWh/100km")
            
    # Nissan Business Parameters
    st.subheader("Business Parameters")
    col3, col4 = st.columns(2)
    with col3:
        nissan_margin = st.slider("Nissan Margin (%)",
                                 min_value=0, max_value=50,
                                 value=20, step=1)
    with col4:
        nissan_volume = st.slider("Nissan Volume (units)",
                                 min_value=1, max_value=5000,
                                 value=500, step=10)

    st.subheader("PV Surface Configuration")

    # Create checkboxes and sliders for each surface
    surfaces_config = {}
    cols = st.columns(3)

    for i, (surface_name, surface_data) in enumerate(segment_data['surfaces'].items()):
        with cols[i % 3]:
            display_name = surface_name.replace('_', ' ').title()
            include = st.checkbox(f"Include {display_name}",
                                 value=surface_data.get('default', True),
                                 key=f"include_{surface_name}")

            if include:
                area = st.number_input(f"{display_name} Area (m²)",
                                      min_value=0.0,
                                      value=surface_data['area'],
                                      step=0.1,
                                      key=f"area_{surface_name}")

                utilization = st.slider(f"{display_name} Utilization (%)",
                                       min_value=0, max_value=100,
                                       value=default_utilization,
                                       key=f"util_{surface_name}")

                angle = st.slider(f"{display_name} Angle (°)",
                                 min_value=0, max_value=90,
                                 value=surface_data['angle'],
                                 key=f"angle_{surface_name}")

                efficiency = st.slider(f"{display_name} PV Efficiency (%)",
                                      min_value=0, max_value=100,
                                      value=default_pv_efficiency,
                                      key=f"eff_{surface_name}")

                cost = st.number_input(f"{display_name} Cost (€/m²)",
                                      min_value=0,
                                      value=default_cost,
                                      key=f"cost_{surface_name}")

                surfaces_config[surface_name] = {
                    'area': area,
                    'utilization': utilization,
                    'angle': angle,
                    'efficiency': efficiency,
                    'cost': cost,
                    'include': True
                }
            else:
                surfaces_config[surface_name] = {'include': False}

    # Other parameters
    st.subheader("Other Parameters")
    transformation_efficiency = st.slider("Energy Transformation Efficiency (%)",
                                         min_value=0, max_value=100,
                                         value=default_transformation_efficiency)

with tab2:
    st.header("Premium Analysis")
    
    if st.button("Calculate Results", type="primary", use_container_width=True):
        # Determine which irradiation data to use
        if irradiation_source in ["Solarcast API Forecast", "Tutiempo 15-Day Forecast"] and daily_irradiation:
            # Use the selected day's irradiation for all months
            irradiation_to_use = [daily_irradiation] * 12
        else:
            # Use the monthly average data
            irradiation_to_use = irradiation_data[region]
            data_source = "Monthly Average"
        
        # Calculate PV energy production for each surface
        total_area = 0
        total_daily_energy = 0
        total_cost = 0
        monthly_energy = {month: 0 for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']}
        surfaces_results = []

        for surface_name, config in surfaces_config.items():
            if config.get('include', False):
                # Calculate for each month
                monthly_surface_energy = []
                for month_idx, month in enumerate(monthly_energy.keys()):
                    # Adjust irradiation by angle (simple cosine correction)
                    angle_rad = np.radians(config['angle'])
                    effective_irradiation = irradiation_to_use[month_idx] * np.cos(angle_rad)

                    # Calculate energy for this month
                    area = config['area'] * (config['utilization']/100)
                    energy = area * effective_irradiation * (config['efficiency']/100) * (transformation_efficiency/100)

                    # For side windows, multiply by 2 (left and right)
                    if 'side' in surface_name:
                        energy *= 2

                    monthly_surface_energy.append(energy)
                    monthly_energy[month] += energy

                # Calculate average daily energy
                avg_daily_energy = np.mean(monthly_surface_energy)

                # Calculate cost (not multiplied for side windows - cost is per panel)
                cost = config['area'] * config['cost']
                if 'side' in surface_name:
                    cost *= 2

                surfaces_results.append({
                    'name': surface_name.replace('_', ' ').title(),
                    'area': config['area'],
                    'effective_area': config['area'] * (config['utilization']/100),
                    'avg_daily_energy': avg_daily_energy,
                    'monthly_energy': monthly_surface_energy,
                    'cost': cost
                })

                total_area += config['area'] * (config['utilization']/100) * (2 if 'side' in surface_name else 1)
                total_daily_energy += avg_daily_energy
                total_cost += cost

        # Calculate average efficiency
        if total_area > 0:
            avg_efficiency = (total_daily_energy / (total_area * np.mean(irradiation_to_use) * (transformation_efficiency/100))) * 100
        else:
            avg_efficiency = 0

        # Calculate ranges
        wltp_range = (total_daily_energy / (segment_data['wltp'] / 100))  # Convert to km
        city_range = (total_daily_energy / (segment_data['city'] / 100))

        # Calculate monthly ranges
        monthly_wltp_range = {month: (energy / (segment_data['wltp'] / 100)) for month, energy in monthly_energy.items()}
        monthly_city_range = {month: (energy / (segment_data['city'] / 100)) for month, energy in monthly_energy.items()}

        # Calculate annual savings and payback period
        annual_energy_kwh = sum(monthly_energy.values()) * 30.44  # Average days per month
        annual_savings = annual_energy_kwh * electricity_price  # Now in EUR
        payback_period = total_cost / annual_savings if annual_savings > 0 else float('inf')

        # Calculate Nissan Annual Profit (in k€)
        nissan_profit = (total_cost * (nissan_margin/100) * nissan_volume) / 1000

        # Display results
        st.subheader("Feasibility Study Summary")
        st.info(f"Using irradiation data: **{data_source}**")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Area", f"{total_area:.2f} m²")
            st.metric("Average Efficiency", f"{avg_efficiency:.1f}%")
            st.metric("Avg. Daily Output", f"{total_daily_energy:.0f} kWh")

        with col2:
            st.metric("Avg. Daily Range (WLTP)", f"{wltp_range:.1f} km")
            st.metric("Avg. Daily Range (City)", f"{city_range:.1f} km")
            st.metric("Annual Energy Production", f"{annual_energy_kwh:.0f} kWh")

        with col3:
            st.metric("Total Investment", f"{total_cost:.0f} €")
            st.metric("Annual Savings", f"{annual_savings:.0f} €")
            st.metric("Payback Period", f"{payback_period:.1f} years" if not np.isinf(payback_period) else "∞")
            
        with col4:
            st.metric("Nissan Volume", f"{nissan_volume} units")
            st.metric("Nissan Margin", f"{nissan_margin}%")
            st.metric("Nissan Annual Profit", f"{nissan_profit:.1f} k€")

        # ---- Modern Visualization 1: Dual Metric Energy Chart ----
        st.subheader("Solar Energy Performance")
        
        # Convert to DataFrame and adjust units
        monthly_df = pd.DataFrame({
            'Month': list(monthly_energy.keys()),
            'Irradiation (kWh/m²/day)': irradiation_to_use,
            'Energy Gain (kWh/day)': [e for e in monthly_energy.values()]
        })

        # Create figure with secondary y-axis
        fig = px.line(monthly_df, 
                    x='Month', 
                    y=['Irradiation (kWh/m²/day)', 'Energy Gain (kWh/day)'],
                    title=f'<b>Monthly Solar Energy Performance ({data_source})</b>',
                    labels={'value': 'Energy (kWh)', 'variable': 'Metric'},
                    color_discrete_sequence=['#FFA15A', '#636EFA'],
                    template='plotly_white')

        # Formatting updates
        fig.update_layout(
            hovermode="x unified",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial", size=12),
            yaxis=dict(
                title='Energy (kWh)',
                tickformat=".1f",
                range=[0, max(monthly_df['Irradiation (kWh/m²/day)'].max(), 
                        monthly_df['Energy Gain (kWh/day)'].max()) * 1.1]
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Customize hover data
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>" +
                        "%{yaxis.title.text}: %{y:.1f} kWh<extra></extra>"
        )

        st.plotly_chart(fig, use_container_width=True)
        
        # ---- Modern Visualization 2: Range Gain Bars ----
        st.subheader("Driving Range Enhancement")
        range_df = pd.DataFrame({
            'Month': list(monthly_energy.keys()),
            'WLTP Range': list(monthly_wltp_range.values()),
            'City Range': list(monthly_city_range.values())
        })
        
        fig2 = px.bar(range_df, x='Month', y=['WLTP Range', 'City Range'],
                     barmode='group',
                     title=f'<b>Additional Daily Driving Range ({data_source})</b>',
                     labels={'value': 'Kilometers', 'variable': 'Cycle'},
                     color_discrete_sequence=['#00CC96', '#AB63FA'])
        
        # Set y-axis ticks to increment by 5km
        max_range = max(max(monthly_wltp_range.values()), max(monthly_city_range.values()))
        fig2.update_layout(
            hovermode="x unified",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial", size=12),
            yaxis=dict(
                tickformat=".1f",
                tickmode='linear',
                tick0=0,
                dtick=5,
                range=[0, max_range * 1.1]
            )
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # ---- Modern Visualization 3: Financial Outlook ----
        st.subheader("Financial Outlook")
        
        # Create timeline (0 to max_years)
        max_years = 10
        years = list(range(0, max_years + 1))

        # Cumulative savings grows each year
        savings = [annual_savings * year for year in years]

        # Investment remains constant (straight line)
        investment = [total_cost] * len(years)

        profit_df = pd.DataFrame({
            'Year': years,
            'Cumulative Savings (€)': savings,
            'Investment (€)': investment
        })

        # Calculate payback year (when savings >= investment)
        payback_year = next((year for year in years if savings[year] >= investment[year]), None)

        fig3 = px.line(profit_df, x='Year', y=['Cumulative Savings (€)', 'Investment (€)'],
                      title='<b>Investment Payback Timeline</b>',
                      color_discrete_sequence=['#19D3F3', '#FF6692'],
                      markers=True)

        if payback_year is not None:
            # Add payback point marker
            payback_value = savings[payback_year]
            fig3.add_annotation(
                x=payback_year,
                y=payback_value,
                text=f"Payback: {payback_year} years",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40
            )
            # Add vertical line at payback point
            fig3.add_vline(x=payback_year, line_dash="dash", 
                          line_color="gray")

        fig3.update_layout(
            hovermode="x unified",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial", size=12),
            yaxis_title="Euros (€)",
            xaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=1
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Customize hover format
        fig3.update_traces(
            hovertemplate="<b>Year %{x}</b><br>%{y:,.0f} €<extra></extra>"
        )

        st.plotly_chart(fig3, use_container_width=True)
        
        # ---- Modern Visualization 4: Surface Contribution ----
        st.subheader("PV Surface Contribution")
        if surfaces_results:
            contrib_df = pd.DataFrame({
                'Surface': [s['name'] for s in surfaces_results],
                'Contribution (%)': [s['avg_daily_energy']/total_daily_energy*100 for s in surfaces_results],
                'Area (m²)': [s['effective_area'] for s in surfaces_results]
            })
            
            fig4 = px.sunburst(contrib_df, path=['Surface'], values='Contribution (%)',
                              color='Area (m²)', color_continuous_scale='Blues',
                              title=f'<b>Energy Contribution by Surface Area ({data_source})</b>')
            
            fig4.update_layout(
                margin=dict(t=40, l=0, r=0, b=0),
                font=dict(family="Arial", size=12)
            )
            st.plotly_chart(fig4, use_container_width=True)
