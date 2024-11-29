#!/usr/bin/env python3

try:
    from compare.msg import EngagementValue
    print(EngagementValue)
except Exception as e:
    print(f"Cannot import: {e}")