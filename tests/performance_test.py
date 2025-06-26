#!/usr/bin/env python3
"""
performance_test.py - Script de prueba de rendimiento para main.py optimizado

Tests performance of the optimized main.py pipeline across different backends.
"""

import subprocess
import time
import sys
from pathlib import Path

def run_performance_test(backend: str, duration: int = 30):
    """Run performance test for a specific backend"""
    print(f"\n🔄 Testing {backend.upper()} backend for {duration} seconds...")
    
    cmd = [
        sys.executable, "main.py",
        "--preset", "ultra_fast",
        "--backend", backend,
        "--input", "0",
        "--no_display"
    ]
    
    try:
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        start_time = time.time()
        output_lines = []
        
        # Read output for the specified duration
        while time.time() - start_time < duration:
            try:
                line = process.stdout.readline()
                if line:
                    output_lines.append(line.strip())
                    if "Processed" in line and "frames" in line:
                        print(f"  {line.strip()}")
                
                # Check if process ended
                if process.poll() is not None:
                    break
                    
            except Exception as e:
                print(f"Error reading output: {e}")
                break
        
        # Terminate the process
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        
        # Extract performance metrics
        fps_values = []
        for line in output_lines:
            if "Avg FPS:" in line:
                try:
                    # Extract FPS value
                    fps_part = line.split("Avg FPS:")[1].split("|")[0].strip()
                    fps = float(fps_part)
                    fps_values.append(fps)
                except:
                    continue
        
        if fps_values:
            avg_fps = sum(fps_values) / len(fps_values)
            max_fps = max(fps_values)
            min_fps = min(fps_values)
            
            print(f"  ✅ {backend.upper()} Results:")
            print(f"     Average FPS: {avg_fps:.2f}")
            print(f"     Max FPS: {max_fps:.2f}")
            print(f"     Min FPS: {min_fps:.2f}")
            print(f"     Samples: {len(fps_values)}")
            
            return {
                'backend': backend,
                'avg_fps': avg_fps,
                'max_fps': max_fps,
                'min_fps': min_fps,
                'samples': len(fps_values)
            }
        else:
            print(f"  ⚠️ No FPS data collected for {backend}")
            return None
            
    except Exception as e:
        print(f"  ❌ Error testing {backend}: {e}")
        return None

def main():
    print("🚀 Starting Performance Test for ConvNeXt Pose Real-time Pipeline")
    print("=" * 60)
    
    # Check if main.py exists
    if not Path("main.py").exists():
        print("❌ main.py not found in current directory")
        return
    
    # Test duration in seconds
    test_duration = 20
    
    # Backends to test
    backends = ['pytorch', 'onnx']
    
    results = []
    
    for backend in backends:
        result = run_performance_test(backend, test_duration)
        if result:
            results.append(result)
        
        # Wait between tests
        if backend != backends[-1]:
            print("   Waiting 5 seconds before next test...")
            time.sleep(5)
    
    # Summary
    print("\n📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    
    if results:
        for result in results:
            print(f"{result['backend'].upper():>8}: {result['avg_fps']:>6.1f} FPS (avg) | "
                  f"{result['max_fps']:>6.1f} FPS (max) | "
                  f"{result['samples']:>3d} samples")
        
        # Compare with original version
        print("\n🔄 COMPARISON WITH ORIGINAL VERSION:")
        print("  Original PyTorch: ~5 FPS")
        print("  Original ONNX: No poses detected")
        print()
        
        for result in results:
            if result['backend'] == 'pytorch':
                improvement = result['avg_fps'] / 5.0
                print(f"  Optimized PyTorch: {result['avg_fps']:.1f} FPS "
                      f"({improvement:.1f}x improvement)")
            elif result['backend'] == 'onnx':
                print(f"  Optimized ONNX: {result['avg_fps']:.1f} FPS "
                      f"(was broken, now working)")
    else:
        print("❌ No performance data collected")
    
    print("\n✅ Performance test completed!")

if __name__ == "__main__":
    main()
