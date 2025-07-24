#!/bin/bash

# Script để setup project structure cho DeepWukong Backend

echo "🚀 Setting up DeepWukong Backend project structure..."

# Tạo thư mục chính
mkdir -p deepwukong-backend
cd deepwukong-backend

# Tạo cấu trúc thư mục
mkdir -p app/{api,core,models,schemas,services}
mkdir -p deepwukong/src
mkdir -p deepwukong/configs  
mkdir -p storage/{uploads,models,results,logs}
mkdir -p tests/{test_api,test_services,fixtures}
mkdir -p scripts

# Tạo các file __init__.py
touch app/__init__.py
touch app/api/__init__.py
touch app/core/__init__.py
touch app/models/__init__.py
touch app/schemas/__init__.py
touch app/services/__init__.py
touch tests/__init__.py
touch tests/test_api/__init__.py
touch tests/test_services/__init__.py

echo "✅ Project structure created successfully!"
echo "📁 Project structure:"
tree . || find . -type d | sed -e "s/[^-][^\/]*\//  |/g" -e "s/|\([^ ]\)/|-\1/"