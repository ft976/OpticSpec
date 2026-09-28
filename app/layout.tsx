import type {Metadata} from 'next';
import {Syne, Plus_Jakarta_Sans, JetBrains_Mono} from 'next/font/google';
import './globals.css';

const syne = Syne({
  subsets: ['latin'],
  weight: ['600', '700'],
  variable: '--font-display',
});

const plusJakarta = Plus_Jakarta_Sans({
  subsets: ['latin'],
  weight: ['400', '500', '600', '700'],
  variable: '--font-sans',
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  weight: ['400', '500', '600'],
  variable: '--font-mono',
});

const appUrl = process.env.APP_URL || 'https://opticspec.app';

export const metadata: Metadata = {
  metadataBase: new URL(appUrl),
  title: 'OpticSpec — AI Image to Prompt Generator',
  description:
    'Upload any image for deep full-frame pixel analysis—capturing characters, expressions, emotions, motion, text, sunlight, and colour encoding into an exact AI prompt.',
  icons: {
    icon: '/icon.svg',
    shortcut: '/icon.svg',
    apple: '/icon.svg',
  },
  openGraph: {
    title: 'OpticSpec — AI Image to Prompt Generator',
    description:
      'Upload any image for deep full-frame pixel analysis—capturing characters, expressions, emotions, motion, text, sunlight, and colour encoding into an exact AI prompt.',
    type: 'website',
    siteName: 'OpticSpec',
    url: appUrl,
  },
  twitter: {
    card: 'summary_large_image',
    title: 'OpticSpec — AI Image to Prompt Generator',
    description:
      'Upload any image for deep full-frame pixel analysis—capturing characters, expressions, emotions, motion, text, sunlight, and colour encoding into an exact AI prompt.',
  },
};

export default function RootLayout({children}: {children: React.ReactNode}) {
  const jsonLd = {
    '@context': 'https://schema.org',
    '@type': 'WebApplication',
    name: 'OpticSpec — AI Image to Prompt Generator',
    applicationCategory: 'DesignApplication',
    operatingSystem: 'All',
    description:
      'Upload any image for deep full-frame pixel analysis—capturing characters, expressions, emotions, motion, text, sunlight, and colour encoding into an exact AI prompt.',
    offers: {
      '@type': 'Offer',
      price: '0',
      priceCurrency: 'USD',
    },
  };

  return (
    <html
      lang="en"
      className={`${syne.variable} ${plusJakarta.variable} ${jetbrainsMono.variable} dark`}
    >
      <head>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{__html: JSON.stringify(jsonLd)}}
        />
      </head>
      <body
        suppressHydrationWarning
        className="bg-[#0B0F17] text-[#F3F4F6] antialiased selection:bg-[#76B900]/30 selection:text-white"
        style={{fontFamily: 'var(--font-sans), sans-serif'}}
      >
        {children}
      </body>
    </html>
  );
}
