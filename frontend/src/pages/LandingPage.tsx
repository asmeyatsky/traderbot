import { Link } from 'react-router-dom';
import {
  ChatBubbleLeftRightIcon,
  BoltIcon,
  ShieldCheckIcon,
  ChartBarSquareIcon,
  LockClosedIcon,
  ServerIcon,
  CheckBadgeIcon,
} from '@heroicons/react/24/outline';
import FeatureCard from '../components/landing/FeatureCard';

const features = [
  { icon: ChatBubbleLeftRightIcon, title: 'AI Chat Co-Pilot', description: 'Ask questions in plain English. Get real-time prices, ML predictions, and trade recommendations — all through conversation.' },
  { icon: BoltIcon, title: 'One-Click Trades', description: 'See a trade recommendation? Confirm it with a single tap. Your linked Alpaca account executes instantly.' },
  { icon: ShieldCheckIcon, title: 'Built-In Risk Controls', description: 'Automated position sizing, drawdown limits, and a circuit breaker that halts trading during extreme volatility.' },
  { icon: ChartBarSquareIcon, title: 'Portfolio Intelligence', description: 'Track performance, allocation, and risk metrics. Ask the AI to explain your portfolio in plain language.' },
];

const steps = [
  { num: '1', title: 'Ask the AI', description: '"What\'s oversold in tech?" — your AI co-pilot scans the market for opportunities.' },
  { num: '2', title: 'Review & Confirm', description: 'The AI shows trade recommendations with reasoning. You decide what to execute.' },
  { num: '3', title: 'Track Performance', description: 'Monitor your portfolio and get real-time alerts. Ask the AI anything about your positions.' },
];

const trust = [
  { icon: LockClosedIcon, title: 'Encrypted Credentials', description: 'Broker API keys are AES-256 encrypted at rest. Never exposed in API responses.' },
  { icon: ServerIcon, title: 'You Stay in Control', description: 'The AI recommends — you confirm. No trades execute without your explicit approval.' },
  { icon: CheckBadgeIcon, title: 'Paper Trading First', description: 'Test with paper trading before going live. No real money until you\'re ready.' },
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
          Your AI Trading Co-Pilot
        </h1>
        <p className="animate-fade-in-delay-1 mx-auto mt-6 max-w-2xl text-lg leading-relaxed text-indigo-200">
          Ask about any stock in plain English. Get ML-powered predictions, news sentiment,
          and trade recommendations — then execute with one click.
        </p>
        <div className="animate-fade-in-delay-2 mt-10 flex flex-col items-center justify-center gap-4 sm:flex-row">
          <Link
            to="/register"
            className="w-full rounded-md bg-indigo-500 px-6 py-3 text-sm font-semibold text-white shadow-md hover:bg-indigo-400 sm:w-auto"
          >
            Start Free
          </Link>
          <Link
            to="/login"
            className="w-full rounded-md border border-indigo-400 px-6 py-3 text-sm font-semibold text-indigo-200 hover:bg-white/5 sm:w-auto"
          >
            Sign In
          </Link>
        </div>

        {/* Chat preview bubble */}
        <div className="animate-fade-in-delay-2 mx-auto mt-12 max-w-md rounded-2xl bg-white/10 p-4 text-left ring-1 ring-white/20 backdrop-blur-sm">
          <div className="flex items-start gap-3">
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-indigo-500 text-xs font-bold text-white">
              AI
            </div>
            <div className="text-sm leading-relaxed text-indigo-100">
              AAPL is trading at <span className="font-semibold text-white">$187.42</span>.
              RSI is at <span className="font-semibold text-white">28.3</span> (oversold).
              Our ML model gives a{' '}
              <span className="font-semibold text-green-400">BUY signal</span> with 78% confidence.
              Want me to place an order?
            </div>
          </div>
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
        <h2 className="text-center text-3xl font-bold text-white">Safe by Design</h2>
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
        <h2 className="text-2xl font-bold text-white">Ready to trade smarter?</h2>
        <p className="mt-3 text-indigo-200">Your AI co-pilot is waiting.</p>
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
