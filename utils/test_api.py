"""
Test API for validating code loading in Colab
用于验证在Colab中加载代码的测试API
"""

import sys
import os
import torch
import datetime

def test_basic_import():
    """Test basic imports and system info"""
    print("🔧 Testing Basic Imports:")
    print(f"  Python version: {sys.version.split()[0]}")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU device: {torch.cuda.get_device_name(0)}")
    print(f"  Current working directory: {os.getcwd()}")
    return True

def test_project_structure():
    """Test if project structure is loaded correctly"""
    print("\n📁 Testing Project Structure:")
    
    # Expected directories
    expected_dirs = ['models', 'data', 'losses', 'utils', 'configs', 'training']
    
    for dir_name in expected_dirs:
        if os.path.exists(dir_name):
            print(f"  ✅ {dir_name}/ found")
        else:
            print(f"  ❌ {dir_name}/ missing")
    
    return all(os.path.exists(d) for d in expected_dirs)

def test_model_imports():
    """Test importing key model components"""
    print("\n🧠 Testing Model Imports:")
    
    try:
        from models.model_pts_vae import SVGTransformer
        print("  ✅ SVGTransformer imported successfully")
        
        from models.config import _DefaultConfig
        print("  ✅ _DefaultConfig imported successfully")
        
        from data.my_svg_dataset_pts import SVGDataset_nopadding
        print("  ✅ SVGDataset_nopadding imported successfully")
        
        return True
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_config_loading():
    """Test loading configuration files"""
    print("\n⚙️ Testing Config Loading:")
    
    try:
        import yaml
        config_path = 'configs/vae_config_cmd_10.yaml'
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"  ✅ Config loaded: {len(config)} parameters")
            print(f"  📋 Key params: dim_z={config.get('dim_z')}, batch_size={config.get('batch_size')}")
            return True
        else:
            print(f"  ❌ Config file not found: {config_path}")
            return False
    except Exception as e:
        print(f"  ❌ Config loading failed: {e}")
        return False

def run_all_tests():
    """Run comprehensive test suite"""
    print("🚀 Starting Dual-Branch VAE Test Suite")
    print("=" * 50)
    print(f"⏰ Test started at: {datetime.datetime.now()}")
    print("=" * 50)
    
    tests = [
        ("Basic Import Test", test_basic_import),
        ("Project Structure Test", test_project_structure), 
        ("Model Import Test", test_model_imports),
        ("Config Loading Test", test_config_loading)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  💥 {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Environment is ready for VAE training.")
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
    
    return passed == len(results)

if __name__ == "__main__":
    run_all_tests()
