# VHS demo recording for whichllm
# Install: https://github.com/charmbracelet/vhs
# Run: vhs demo.tape
# Output: assets/demo.gif

Output assets/demo.gif

Set FontSize 15
Set Width 1200
Set Height 700
Set Theme "Catppuccin Mocha"
Set Padding 20
Set TypingSpeed 40ms

# Title
Type "# whichllm - Find the best LLM for your hardware"
Enter
Sleep 1s

# Basic usage with GPU simulation
Type "whichllm --gpu 'RTX 4090' --top 5"
Sleep 300ms
Enter
Sleep 4s

# Plan command - reverse lookup
Type "whichllm plan 'llama 3 70b' --gpu 'RTX 4090'"
Sleep 300ms
Enter
Sleep 4s

# Profile-based search
Type "whichllm --gpu 'RTX 4090' --profile coding --top 3"
Sleep 300ms
Enter
Sleep 3s

Sleep 2s
