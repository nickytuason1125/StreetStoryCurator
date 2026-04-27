import re

# Read the original file
with open('street-story-curator/src/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add "Strong ✅ + Mid ⚠️ + Weak ❌" to the filter dropdown options
# Find the filter dropdown choices and add the new option
pattern = r'(choices=\["All", "Strong ✅", "Mid ⚠️", "Weak ❌", "Strong ✅ \+ Mid ⚠️", "Strong ✅ \+ Weak ❌", "Mid ⚠️ \+ Weak ❌", "All Grades")'
replacement = r'choices=["All", "Strong ✅", "Mid ⚠️", "Weak ❌", "Strong ✅ + Mid ⚠️", "Strong ✅ + Weak ❌", "Mid ⚠️ + Weak ❌", "Strong ✅ + Mid ⚠️ + Weak ❌", "All Grades"]'
content = re.sub(pattern, replacement, content)

# 2. Add a new condition in the filter function to handle the new filter option
# Find the filter function and add the new condition
filter_pattern = r'(\s*elif filter_grade == "All Grades":\s*filtered = \[r for r in rows if len\(r\) > 1 and \("Strong" in str\(r\[1\]\) or "Mid" in str\(r\[1\]\) or "Weak" in str\(r\[1\]\)\]\s*)'
filter_replacement = r'''
    elif filter_grade == "Strong ✅ + Mid ⚠️ + Weak ❌":
        filtered = [r for r in rows if len(r) > 1 and ("Strong" in str(r[1]) or "Mid" in str(r[1]) or "Weak" in str(r[1]))]
    elif filter_grade == "All Grades":
        filtered = [r for r in rows if len(r) > 1 and ("Strong" in str(r[1]) or "Mid" in str(r[1]) or "Weak" in str(r[1]))]
'''
content = re.sub(filter_pattern, filter_replacement, content)

# Write the modified content back to the file
with open('street-story-curator/src/app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("App.py has been successfully modified!")