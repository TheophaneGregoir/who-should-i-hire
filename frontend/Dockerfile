# Use an official Node.js image as a build stage
FROM node:18-alpine AS build

# Set working directory
WORKDIR /app

# Copy dependency files first to leverage Docker cache
COPY package.json package-lock.json ./

# Install dependencies
RUN npm install

# Copy the rest of the app's source code
COPY . .

# Build the frontend app (output goes to `dist`)
RUN npm run build

# Stage 2: Serve the app with a minimal web server
FROM node:18-alpine

# Install 'serve' to host the production build
RUN npm install -g serve

# Copy the built app from the previous stage
COPY --from=build /app/dist /app/dist

# Set working directory
WORKDIR /app

# Expose the port serve will use
EXPOSE 3000

# Start the app
CMD ["serve", "-s", "dist", "-l", "3000"]
