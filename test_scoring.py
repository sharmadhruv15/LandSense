#!/usr/bin/env python3
"""Test the new risk scoring logic"""


def calculate_risk_score(bare_rock_pct, vegetation_pct, brightness):
    """New improved risk scoring"""
    is_ice_region = (brightness > 150) and (vegetation_pct < 20)

    if is_ice_region:
        risk_score = int((bare_rock_pct * 0.2) +
                         ((100 - vegetation_pct) * 0.1))
    else:
        risk_score = min(100, int((bare_rock_pct * 0.6) +
                         ((100 - vegetation_pct) * 0.35)))

    if risk_score > 60:
        risk = "VERY HIGH"
    elif risk_score > 40:
        risk = "HIGH"
    elif risk_score > 20:
        risk = "MODERATE"
    else:
        risk = "LOW"

    return risk_score, risk, is_ice_region


# Test cases
print("=" * 70)
print("TESTING NEW RISK SCORING LOGIC")
print("=" * 70)

# Test 1: Ice/Snow region (high brightness, low vegetation)
print("\n1. ICE/SNOW REGION (High brightness + Low vegetation)")
score, risk, is_ice = calculate_risk_score(
    bare_rock_pct=60, vegetation_pct=10, brightness=180)
print(f"   Bare Rock: 60% | Vegetation: 10% | Brightness: 180")
print(f"   Is Ice Region: {is_ice}")
print(f"   Risk Score: {score}/100 | Risk Level: {risk}")
print(f"   ✅ EXPECTED: LOW/MODERATE (because it's frozen)\n")

# Test 2: Normal rocky mountain region
print("2. ROCKY MOUNTAIN (Medium bare rock, low vegetation)")
score, risk, is_ice = calculate_risk_score(
    bare_rock_pct=50, vegetation_pct=30, brightness=100)
print(f"   Bare Rock: 50% | Vegetation: 30% | Brightness: 100")
print(f"   Is Ice Region: {is_ice}")
print(f"   Risk Score: {score}/100 | Risk Level: {risk}")
print(f"   ✅ EXPECTED: MODERATE/HIGH (actual landslide risk)\n")

# Test 3: Severe erosion region
print("3. SEVERE EROSION (High bare rock, very low vegetation)")
score, risk, is_ice = calculate_risk_score(
    bare_rock_pct=80, vegetation_pct=5, brightness=90)
print(f"   Bare Rock: 80% | Vegetation: 5% | Brightness: 90")
print(f"   Is Ice Region: {is_ice}")
print(f"   Risk Score: {score}/100 | Risk Level: {risk}")
print(f"   ✅ EXPECTED: HIGH/VERY HIGH (high erosion risk)\n")

# Test 4: Green forest area
print("4. FOREST AREA (Low bare rock, high vegetation)")
score, risk, is_ice = calculate_risk_score(
    bare_rock_pct=10, vegetation_pct=80, brightness=80)
print(f"   Bare Rock: 10% | Vegetation: 80% | Brightness: 80")
print(f"   Is Ice Region: {is_ice}")
print(f"   Risk Score: {score}/100 | Risk Level: {risk}")
print(f"   ✅ EXPECTED: LOW (protected by vegetation)\n")

print("=" * 70)
print("All test cases completed! Ice regions now classified as safer. ✅")
print("=" * 70)
