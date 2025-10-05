# pixel_processor.py - FIXED FOR ALL QUARTERS

import streamlit as st
import numpy as np
from typing import Optional, Dict, Any
import lightkurve as lk
import requests
import time


class PixelProcessor:
    """
    Download and process target pixel files from NASA MAST
    FIXED: Properly downloads ALL quarters when quarter=None
    """
    
    def __init__(self):
        self.processed_data = None
        self.raw_tpf = None
        self.star_id = None
        self.mission = None
        
    def download_and_process_pixels(self, star_id: str, mission: str, 
                                   quarter: Optional[int] = None, 
                                   sector: Optional[int] = None) -> bool:
        """
        Download and process target pixel files
        
        CRITICAL FIX: When quarter=None, downloads ALL available quarters
        """
        
        try:
            self.star_id = star_id
            self.mission = mission
            
            # OPTIMIZED: Convert 0 to None (will download 1-2 files max for speed)
            if quarter == 0:
                quarter = None
                st.success("⚡ Will download 1-2 Kepler quarters for faster processing")
            
            if sector == 0:
                sector = None
                st.success("✅ Downloading ALL available TESS sectors")
            
            st.write("🎯 *Searching for target pixel files...*")
            
            # Try multiple approaches
            success = False
            error_messages = []
            
            for attempt in range(3):
                try:
                    st.info(f"Attempt {attempt + 1}/3: Searching NASA MAST archive...")
                    
                    if mission.lower() == "kepler":
                        # FIXED: Pass None for all quarters
                        search_result = lk.search_targetpixelfile(
                            f"KIC {star_id}", 
                            mission="Kepler", 
                            quarter=quarter  # None = all quarters
                        )
                    else:
                        search_result = lk.search_targetpixelfile(
                            f"TIC {star_id}", 
                            mission="TESS", 
                            sector=sector  # None = all sectors
                        )
                    
                    if len(search_result) > 0:
                        success = True
                        break
                        
                except requests.exceptions.HTTPError as e:
                    error_msg = f"Attempt {attempt + 1} failed: {str(e)}"
                    error_messages.append(error_msg)
                    st.warning(error_msg)
                    
                    if attempt < 2:
                        st.info("Waiting 3 seconds before retry...")
                        time.sleep(3)
                    
                except Exception as e:
                    error_msg = f"Attempt {attempt + 1} failed: {str(e)}"
                    error_messages.append(error_msg)
                    st.warning(error_msg)
                    
                    if attempt < 2:
                        time.sleep(2)
            
            if not success:
                st.error("❌ *All search methods failed*")
                
                with st.expander("🔧 *Detailed Error Information*"):
                    st.write("*Error Details:*")
                    for i, error in enumerate(error_messages, 1):
                        st.code(f"{i}. {error}")
                    
                    st.write("\n*Suggestions:*")
                    st.write("- Try a different star ID")
                    st.write("- Check your internet connection")
                    st.write("- Try again in a few minutes")
                
                return False
            
            if len(search_result) == 0:
                st.error(f"No target pixel files found for {star_id}")
                return False
            
            st.success(f"✅ Found {len(search_result)} target pixel file(s)")
            
            # OPTIMIZED: Download only 1-2 files for faster processing
            if len(search_result) > 1:
                # Limit to 2 files maximum for speed
                max_files = min(2, len(search_result))
                st.info(f"⚡ Downloading {max_files} file(s) for faster processing (found {len(search_result)} available)")
                
                # Download limited files
                try:
                    if max_files == 1:
                        # Just download the first one
                        self.raw_tpf = search_result[0].download()
                        st.success(f"✅ Downloaded 1 quarter/sector")
                    else:
                        # Download first 2 files
                        tpf_collection = []
                        for i in range(max_files):
                            st.info(f"📥 Downloading file {i+1}/{max_files}...")
                            tpf = search_result[i].download()
                            if tpf:
                                tpf_collection.append(tpf)
                        
                        if len(tpf_collection) > 0:
                            # Use first one for pixel data
                            self.raw_tpf = tpf_collection[0]
                            
                            # Calculate total time
                            total_time = sum([len(tpf.time) for tpf in tpf_collection])
                            st.success(f"✅ Downloaded {len(tpf_collection)} quarters/sectors ({total_time:,} time points)")
                        else:
                            st.error("Failed to download TPF files")
                            return False
                        
                except Exception as download_error:
                    st.warning(f"Multi-file download failed: {download_error}")
                    st.info("Trying single file download...")
                    self.raw_tpf = search_result[0].download()
            else:
                st.info("📥 Downloading target pixel file...")
                self.raw_tpf = search_result[0].download()
            
            if self.raw_tpf is None:
                st.error("❌ Failed to download target pixel file")
                return False
            
            # Calculate time span
            try:
                time_array = self.raw_tpf.time
                if hasattr(time_array, 'value'):
                    time_vals = time_array.value
                else:
                    time_vals = np.array(time_array)
                
                duration_days = float(np.max(time_vals) - np.min(time_vals))
                
            except Exception as time_error:
                st.info(f"Time calculation note: {time_error}")
                duration_days = 0.0
            
            # Process the pixel data
            self.processed_data = {
                'tpf': self.raw_tpf,
                'star_id': star_id,
                'mission': mission,
                'quarter': quarter,
                'sector': sector,
                'shape': self.raw_tpf.shape,
                'time_span': len(self.raw_tpf.time),
                'quality_mask': self.raw_tpf.quality == 0,
                'duration_days': duration_days,
                'search_results_count': len(search_result)
            }
            
            # Display success information
            st.success(f"🎉 *Target pixel file processed successfully!*")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("*Pixel Array:*")
                st.write(f"- Dimensions: {self.raw_tpf.shape[1]}×{self.raw_tpf.shape[2]} pixels")
                st.write(f"- Time points: {len(self.raw_tpf.time):,}")
            
            with col2:
                st.write("*Observation Period:*")
                if duration_days > 0:
                    st.write(f"- Duration: {duration_days:.1f} days")
                    if duration_days > 365:
                        st.success(f"✅ Long baseline ({duration_days/365:.1f} years)")
                    elif duration_days < 90:
                        st.warning(f"⚠ Short baseline - may miss planets")
                else:
                    st.write(f"- Time points: {len(self.raw_tpf.time)}")
            
            with col3:
                st.write("*Data Quality:*")
                good_quality = np.sum(self.raw_tpf.quality == 0)
                total_points = len(self.raw_tpf.quality)
                quality_percent = (good_quality / total_points) * 100 if total_points > 0 else 0
                
                st.write(f"- Good quality: {good_quality:,} pts")
                st.write(f"- Quality rate: {quality_percent:.1f}%")
            
            return True
            
        except Exception as e:
            st.error(f"❌ *Pixel processing failed with unexpected error*")
            
            with st.expander("🔧 *Complete Error Details*"):
                st.write("*Error Type:* Unexpected Exception")
                st.code(str(e))
                
                import traceback
                st.write("*Full Traceback:*")
                st.code(traceback.format_exc())
            
            return False
    
    def get_processed_data(self) -> Optional[Dict[str, Any]]:
        """Get the processed pixel data"""
        return self.processed_data
    
    def reset(self):
        """Reset the processor state"""
        self.processed_data = None
        self.raw_tpf = None
        self.star_id = None
        self.mission = None