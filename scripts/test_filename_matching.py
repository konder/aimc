#!/usr/bin/env python3
"""
测试网盘文件名匹配功能

测试网盘转存后的特殊字符替换是否能正确匹配。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.clip4mc_data_pipeline import (
    normalize_netdisk_filename,
    find_video_file,
    build_file_index
)


def test_netdisk_normalization():
    """测试网盘文件名规范化"""
    print("=" * 60)
    print("测试网盘文件名规范化")
    print("=" * 60)
    
    test_cases = [
        # (网盘文件名, 期望的原始文件名)
        ("What If All SCPs Combined Together？ ｜ Minecraft Roleplay.mp4", 
         "What If All SCPs Combined Together? | Minecraft Roleplay.mp4"),
        
        ("Westside Roommates EP2 ⧸⧸ Cookies, Milk, and Murder？! {MINECRAFT ROLEPLAY}.mp4",
         "Westside Roommates EP2 // Cookies, Milk, and Murder?! {MINECRAFT ROLEPLAY}.mp4"),
        
        ("What&#39;s New in Minecraft Snapshots 15w32a, 15w32b and 15w32c？.mp4",
         "What's New in Minecraft Snapshots 15w32a, 15w32b and 15w32c?.mp4"),
        
        ("WORLD&#39;S LARGEST MINECRAFT ILLUSION!.mp4",
         "WORLD'S LARGEST MINECRAFT ILLUSION!.mp4"),
        
        ("What if Minecraft 1.14 had SWAMP VILLAGES？ ： TIMELAPSE + DOWNLOAD.mp4",
         "What if Minecraft 1.14 had SWAMP VILLAGES? : TIMELAPSE + DOWNLOAD.mp4"),
    ]
    
    all_passed = True
    for netdisk_name, expected_original in test_cases:
        result = normalize_netdisk_filename(netdisk_name)
        passed = result == expected_original
        all_passed = all_passed and passed
        
        status = "✅" if passed else "❌"
        print(f"\n{status} 测试:")
        print(f"  网盘: {netdisk_name}")
        print(f"  期望: {expected_original}")
        print(f"  结果: {result}")
        if not passed:
            print(f"  差异: 期望和结果不匹配")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有测试通过")
    else:
        print("❌ 部分测试失败")
    print("=" * 60)
    
    return all_passed


def test_file_matching(videos_dir: Path):
    """测试文件匹配功能"""
    print("\n" + "=" * 60)
    print(f"测试文件匹配: {videos_dir}")
    print("=" * 60)
    
    if not videos_dir.exists():
        print(f"❌ 目录不存在: {videos_dir}")
        return False
    
    # 建立文件索引
    all_files = build_file_index(videos_dir)
    print(f"找到 {len(all_files)} 个视频文件")
    
    # 测试用例：原始文件名（带特殊字符）
    test_filenames = [
        "What If All SCPs Combined Together? | Minecraft Roleplay",
        "Westside Roommates EP2 // Cookies, Milk, and Murder?! {MINECRAFT ROLEPLAY}",
        "What's New in Minecraft Snapshots 15w32a, 15w32b and 15w32c?",
        "WORLD'S LARGEST MINECRAFT ILLUSION!",
        "What if Minecraft 1.14 had SWAMP VILLAGES? : TIMELAPSE + DOWNLOAD",
    ]
    
    print("\n测试匹配:")
    matched_count = 0
    for filename in test_filenames:
        result = find_video_file(videos_dir, filename, all_files)
        if result:
            matched_count += 1
            print(f"✅ 匹配: {filename}")
            print(f"   → {result.name}")
        else:
            print(f"❌ 未匹配: {filename}")
    
    print("\n" + "=" * 60)
    print(f"匹配结果: {matched_count}/{len(test_filenames)}")
    print("=" * 60)
    
    return matched_count == len(test_filenames)


def analyze_unmatched_files(videos_dir: Path, info_csv: Path):
    """分析未匹配的文件"""
    print("\n" + "=" * 60)
    print("分析未匹配文件")
    print("=" * 60)
    
    if not videos_dir.exists():
        print(f"❌ 目录不存在: {videos_dir}")
        return
    
    if not info_csv.exists():
        print(f"❌ CSV文件不存在: {info_csv}")
        return
    
    from src.utils.clip4mc_data_pipeline import parse_info_csv
    
    # 解析 CSV
    vid_to_filename = parse_info_csv(info_csv)
    print(f"CSV 记录: {len(vid_to_filename)} 条")
    
    # 建立文件索引
    all_files = build_file_index(videos_dir)
    print(f"视频文件: {len(all_files)} 个")
    
    # 检查匹配情况
    matched = 0
    unmatched = []
    
    for vid, filename in vid_to_filename.items():
        result = find_video_file(videos_dir, filename, all_files)
        if result:
            matched += 1
        else:
            unmatched.append((vid, filename))
    
    print(f"\n匹配情况:")
    print(f"  已匹配: {matched}/{len(vid_to_filename)} ({matched/len(vid_to_filename)*100:.1f}%)")
    print(f"  未匹配: {len(unmatched)}")
    
    if unmatched:
        print(f"\n前 20 个未匹配文件:")
        for i, (vid, filename) in enumerate(unmatched[:20], 1):
            print(f"  {i}. {filename} (vid: {vid})")
    
    print("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="测试网盘文件名匹配")
    parser.add_argument("--test-norm", action='store_true', help="测试规范化函数")
    parser.add_argument("--test-match", type=str, help="测试文件匹配（提供视频目录）")
    parser.add_argument("--analyze", action='store_true', help="分析未匹配文件")
    parser.add_argument("--videos-dir", type=str, help="视频目录")
    parser.add_argument("--info-csv", type=str, help="info.csv 文件")
    
    args = parser.parse_args()
    
    if args.test_norm:
        test_netdisk_normalization()
    
    if args.test_match:
        test_file_matching(Path(args.test_match))
    
    if args.analyze:
        if not args.videos_dir or not args.info_csv:
            print("❌ --analyze 需要 --videos-dir 和 --info-csv 参数")
            return 1
        analyze_unmatched_files(Path(args.videos_dir), Path(args.info_csv))
    
    if not any([args.test_norm, args.test_match, args.analyze]):
        print("使用方法:")
        print("  # 测试规范化")
        print("  python scripts/test_filename_matching.py --test-norm")
        print()
        print("  # 测试文件匹配")
        print("  python scripts/test_filename_matching.py --test-match /path/to/videos")
        print()
        print("  # 分析未匹配文件")
        print("  python scripts/test_filename_matching.py --analyze \\")
        print("      --videos-dir /root/autodl-tmp/TEST2000 \\")
        print("      --info-csv /root/autodl-tmp/test2000.csv")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

