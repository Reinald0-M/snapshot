#!/bin/bash
# Script to push code to GitHub repository

cd "$(dirname "$0")"

# Remove existing .git if corrupted
rm -rf .git

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: PNPS-based nanopore array simulator

- Complete package structure with all core modules
- Langevin dynamics integrator for 2D particle transport
- Multi-pore array geometry and electrostatics
- Visualization and analysis tools
- Example scripts for validation and parameter scanning"

# Add remote repository
git remote add origin https://github.com/Reinald0-M/snapshot.git

# Set main branch
git branch -M main

# Push to GitHub
git push -u origin main

echo "Done! Code has been pushed to https://github.com/Reinald0-M/snapshot"
