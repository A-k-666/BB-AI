"""
Video Downloader Module
=======================
Downloads YouTube videos using yt-dlp.
"""

import yt_dlp
import os


class VideoDownloader:
    """Download YouTube videos"""
    
    def __init__(self):
        pass
    
    def download_youtube(self, url: str, output_path: str = './temp_video.mp4') -> str:
        """
        Download YouTube video
        
        Returns: Path to downloaded video
        """
        
        print("Step 0/4: Downloading YouTube video...")
        
        ydl_opts = {
            'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
            'outtmpl': output_path,
            'quiet': True,
            'no_warnings': True,
            'merge_output_format': 'mp4'
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            print(f"   Downloaded: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"   Download failed: {e}")
            return None


if __name__ == "__main__":
    downloader = VideoDownloader()
    downloader.download_youtube(
        "https://www.youtube.com/watch?v=kKfXj8A4AZA",
        "./test_video.mp4"
    )

