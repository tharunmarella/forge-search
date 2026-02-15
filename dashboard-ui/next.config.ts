/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/traces/:path*',
        destination: process.env.NEXT_PUBLIC_API_URL 
          ? `${process.env.NEXT_PUBLIC_API_URL}/traces/:path*`
          : 'http://localhost:8080/traces/:path*',
      },
    ];
  },
};

export default nextConfig;
