#!/bin/bash

# Script to format Python files using Black in src directory
# Usage: bash format.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Log file
LOG_FILE="/app/src/format.log"

# Function to check if black is installed
install_black() {
    if ! command -v black &> /dev/null; then
        echo -e "${YELLOW}Installing black...${NC}"
        pip install black
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to install black. Please install it manually.${NC}"
            exit 1
        fi
    fi
}

# Function to find src directory
find_src_directory() {
    # First check if we're in src
    if [[ "${PWD##*/}" == "src" ]]; then
        echo "$PWD"
        return 0
    fi
    
    # Then check if we're in a subdirectory of src
    if [[ "$PWD" == */src/* ]]; then
        echo "${PWD%/src/*}/src"
        return 0
    fi
    
    # If not found, exit with error
    echo -e "${RED}Error: Cannot find src directory. Please run this script from within the src directory or its subdirectories${NC}"
    exit 1
}

# Function to format Python files
format_python() {
    local src_dir="$(find_src_directory)"
    echo -e "${YELLOW}Formatting Python files in: ${src_dir}${NC}"
    
    # Move to src directory and find all Python files
    cd "$src_dir"
    
    # Count total Python files
    local total_files=$(find . -type f -name "*.py" -not -path "*/\.*" | wc -l)
    echo -e "${YELLOW}Found ${total_files} Python files to format${NC}"
    
    # Format files
    find . -type f -name "*.py" -not -path "*/\.*" -exec black {} \; 2>> "$LOG_FILE"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Successfully formatted all Python files in src directory${NC}"
    else
        echo -e "${RED}Some files could not be formatted. Check $LOG_FILE for details${NC}"
    fi
}

main() {
    echo -e "${YELLOW}Starting Python code formatting...${NC}"
    
    # Clear previous log
    > "$LOG_FILE"
    
    # Install black if needed
    install_black
    
    # Format Python files
    format_python
    
    # Check if there were any errors
    if [ -s "$LOG_FILE" ]; then
        echo -e "${YELLOW}Some formatting operations generated warnings/errors. Check $LOG_FILE for details.${NC}"
    else
        echo -e "${GREEN}All formatting operations completed successfully!${NC}"
    fi
}

main