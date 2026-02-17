import { Link } from 'react-router-dom';
import {
  SparklesIcon,
  ShieldCheckIcon,
  ChartBarSquareIcon,
  PresentationChartLineIcon,
  LockClosedIcon,
  ServerIcon,
  CheckBadgeIcon,
} from '@heroicons/react/24/outline';
import FeatureCard from '../components/landing/FeatureCard';

const features = [
  { icon: SparklesIcon, title: 'AI Predictions', description: 'Machine learning models analyze market patterns and sentiment to generate actionable trading signals.' },
  { icon: ShieldCheckIcon, title: 'Risk Management', description: 'Automated position sizing, stop-losses, and portfolio-level risk controls to protect your capital.' },
  { icon: ChartBarSquareIcon, title: 'Market Intelligence', description: 'Real-time market data, technical indicators, and news sentiment analysis in one unified view.' },
  { icon: PresentationChartLineIcon, title: 'Portfolio Analytics', description: 'Track performance, allocation, and risk metrics with professional-grade analytics dashboards.' },
];

const steps = [
  { num: '1', title: 'Create Account', description: 'Sign up in under a minute with your risk profile and investment goals.' },
  { num: '2', title: 'Fund & Explore', description: 'Add funds to your account and explore AI-powered market predictions.' },
  { num: '3', title: 'Trade with Confidence', description: 'Place trades backed by data-driven insights and automated risk controls.' },
];

const trust = [
  { icon: LockClosedIcon, title: 'Bank-Grade Encryption', description: 'Your data is protected with industry-standard encryption at rest and in transit.' },
  { icon: ServerIcon, title: 'Real-Time Monitoring', description: 'Systems are monitored around the clock to ensure uptime and reliability.' },
  { icon: CheckBadgeIcon, title: 'Data Protection', description: 'Strict access controls and privacy policies safeguard your information.' },
];

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-950 via-indigo-900 to-slate-900">
      {/* Nav */}
      <header className="flex items-center justify-between px-6 py-4 lg:px-12">
        <span className="text-xl font-bold text-white">TraderBot</span>
        <div className="flex items-center gap-4">
          <Link to="/login" className="text-sm font-medium text-indigo-200 hover:text-white">
            Sign In
          </Link>
          <Link
            to="/register"
            className="rounded-md bg-indigo-500 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-400"
          >
            Get Started
          </Link>
        </div>
      </header>

      {/* Hero */}
      <section className="mx-auto max-w-4xl px-6 py-20 text-center lg:py-32">
        <h1 className="animate-fade-in text-4xl font-extrabold tracking-tight text-white sm:text-5xl lg:text-6xl">
          Trade Smarter with AI-Powered Intelligence
        </h1>
        <p className="animate-fade-in-delay-1 mx-auto mt-6 max-w-2xl text-lg leading-relaxed text-indigo-200">
          Harness machine learning, real-time market data, and automated risk management to make
          confident, data-driven trading decisions.
        </p>
        <div className="animate-fade-in-delay-2 mt-10 flex items-center justify-center gap-4">
          <Link
            to="/register"
            className="rounded-md bg-indigo-500 px-6 py-3 text-sm font-semibold text-white shadow-md hover:bg-indigo-400"
          >
            Get Started Free
          </Link>
          <Link
            to="/login"
            className="rounded-md border border-indigo-400 px-6 py-3 text-sm font-semibold text-indigo-200 hover:bg-white/5"
          >
            Sign In
          </Link>
        </div>
      </section>

      {/* Features */}
      <section className="mx-auto max-w-6xl px-6 pb-20">
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {features.map((f) => (
            <FeatureCard key={f.title} {...f} />
          ))}
        </div>
      </section>

      {/* How It Works */}
      <section className="mx-auto max-w-4xl px-6 py-20 text-center">
        <h2 className="text-3xl font-bold text-white">How It Works</h2>
        <div className="mt-12 grid grid-cols-1 gap-8 sm:grid-cols-3">
          {steps.map((s) => (
            <div key={s.num}>
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-full bg-indigo-500 text-lg font-bold text-white">
                {s.num}
              </span>
              <h3 className="mt-4 text-lg font-semibold text-white">{s.title}</h3>
              <p className="mt-2 text-sm text-indigo-200">{s.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Trust */}
      <section className="mx-auto max-w-5xl px-6 py-20">
        <h2 className="text-center text-3xl font-bold text-white">Built for Security</h2>
        <div className="mt-12 grid grid-cols-1 gap-6 sm:grid-cols-3">
          {trust.map((t) => (
            <div key={t.title} className="rounded-xl bg-white/5 p-6 text-center ring-1 ring-white/10">
              <t.icon className="mx-auto h-8 w-8 text-indigo-400" />
              <h3 className="mt-4 text-base font-semibold text-white">{t.title}</h3>
              <p className="mt-2 text-sm text-indigo-200">{t.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Footer CTA */}
      <section className="bg-indigo-600/20 py-16 text-center">
        <h2 className="text-2xl font-bold text-white">Ready to start trading smarter?</h2>
        <p className="mt-3 text-indigo-200">Join TraderBot and let AI work for your portfolio.</p>
        <Link
          to="/register"
          className="mt-8 inline-block rounded-md bg-indigo-500 px-8 py-3 text-sm font-semibold text-white shadow-md hover:bg-indigo-400"
        >
          Create Free Account
        </Link>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/10 px-6 py-8 text-center text-xs text-indigo-300">
        &copy; {new Date().getFullYear()} TraderBot. All rights reserved.
      </footer>
    </div>
  );
}
