# Product Overview

This is an end-to-end machine learning project for predicting late shipments in supply chain management. The system helps a global sports and outdoor equipment retailer proactively identify high-risk shipments before delays occur.

## Core Functionality

- **Late Shipment Prediction**: Classifies shipments that will be delayed by 1+ days (86.14% accuracy)
- **Very Late Shipment Prediction**: Identifies shipments delayed by 3+ days (97.58% recall)
- **REST API**: FastAPI-based service for real-time predictions
- **Interactive Dashboard**: Web-based geospatial visualization for monitoring shipment risks

## Business Value

The dataset shows 57% of shipments are late by at least one day, with 7% delayed by 3+ days. The ML models enable:
- Proactive intervention for high-risk shipments
- Improved on-time delivery rates
- Reduced customer dissatisfaction and operational costs
- Data-driven logistics optimization

## Deployment

- Containerized FastAPI application deployed on Render
- Interactive Swagger UI at `/docs` endpoint
- Health monitoring via `/ping` endpoint
- Memory-optimized for cloud deployment constraints